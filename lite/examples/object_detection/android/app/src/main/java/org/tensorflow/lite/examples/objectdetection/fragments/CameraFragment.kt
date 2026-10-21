/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 * See https://www.apache.org/licenses/LICENSE-2.0
 */

package org.tensorflow.lite.examples.objectdetection.fragments

import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(), ObjectDetectorHelper.DetectorListener {

    companion object {
        private const val TAG = "ObjectDetection"
        private const val PI_HOST = "10.242.215.42"
        private const val PI_PORT = 5001
        private const val CONNECT_TIMEOUT_MS = 3000
        private const val SOCKET_TIMEOUT_MS = 5000
        private const val MAX_AUDIO_SIZE = 10 * 1024 * 1024
        private const val REQUIRED_STABLE_FRAMES = 3
        private const val MIN_ANNOUNCEMENT_GAP_MS = 2000L
        private const val SAME_LABEL_REPEAT_DELAY_MS = 6000L
        private const val SPEECH_CONFIDENCE_THRESHOLD = 0.60f
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding: FragmentCameraBinding
        get() = _fragmentCameraBinding
            ?: throw IllegalStateException("Camera binding is unavailable")

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var audioExecutor: ExecutorService

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private var textToSpeech: TextToSpeech? = null
    private var ttsReady = false
    @Volatile private var speechPending = false
    private val audioFiles = ConcurrentHashMap<String, File>()

    private var candidateLabel = ""
    private var candidateCount = 0
    private var lastAnnouncedLabel = ""
    private var lastAnnouncementTime = 0L
    private var lastLabelAnnouncementTime = 0L
    private var utteranceCounter = 0L

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)
        return fragmentCameraBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()
        audioExecutor = Executors.newSingleThreadExecutor()

        fragmentCameraBinding.viewFinder.implementationMode =
            androidx.camera.view.PreviewView.ImplementationMode.COMPATIBLE

        initializeTextToSpeech()

        objectDetectorHelper = ObjectDetectorHelper(
            context = requireContext(),
            objectDetectorListener = this
        )

        fragmentCameraBinding.viewFinder.post {
            if (_fragmentCameraBinding != null) setUpCamera()
        }
    }

    private fun initializeTextToSpeech() {
        textToSpeech = TextToSpeech(requireContext()) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech?.setLanguage(Locale.US)
                ttsReady = result != TextToSpeech.LANG_MISSING_DATA &&
                    result != TextToSpeech.LANG_NOT_SUPPORTED

                textToSpeech?.setOnUtteranceProgressListener(
                    object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            Log.d(TAG, "Speech started: $utteranceId")
                        }

                        override fun onDone(utteranceId: String?) {
                            if (utteranceId != null) {
                                audioFiles.remove(utteranceId)?.let {
                                    sendAudioToRaspberryPi(it)
                                }
                            }
                            speechPending = false
                        }

                        @Deprecated("Deprecated in Java")
                        override fun onError(utteranceId: String?) {
                            if (utteranceId != null) {
                                audioFiles.remove(utteranceId)?.delete()
                            }
                            speechPending = false
                            Log.e(TAG, "Speech synthesis failed")
                        }

                        override fun onError(utteranceId: String?, errorCode: Int) {
                            if (utteranceId != null) {
                                audioFiles.remove(utteranceId)?.delete()
                            }
                            speechPending = false
                            Log.e(TAG, "Speech synthesis error: $errorCode")
                        }
                    }
                )
                Log.d(TAG, "Text-to-Speech initialized")
            } else {
                ttsReady = false
                Log.e(TAG, "Text-to-Speech initialization failed")
            }
        }
    }

    private fun setUpCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(requireContext())
        providerFuture.addListener({
            if (_fragmentCameraBinding == null) return@addListener

            cameraProvider = providerFuture.get()

            preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .build()

            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(cameraExecutor) { image ->
                        if (_fragmentCameraBinding == null) {
                            image.close()
                        } else {
                            detectObjects(image)
                        }
                    }
                }

            try {
                cameraProvider?.unbindAll()
                camera = cameraProvider?.bindToLifecycle(
                    viewLifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
                preview?.setSurfaceProvider(
                    fragmentCameraBinding.viewFinder.surfaceProvider
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
                showToast("Unable to start camera")
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun detectObjects(image: ImageProxy) {
        try {
            if (!::objectDetectorHelper.isInitialized) return

            if (!::bitmapBuffer.isInitialized ||
                bitmapBuffer.width != image.width ||
                bitmapBuffer.height != image.height
            ) {
                bitmapBuffer = Bitmap.createBitmap(
                    image.width,
                    image.height,
                    Bitmap.Config.ARGB_8888
                )
            }

            image.planes[0].buffer.rewind()
            bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)
            objectDetectorHelper.detect(bitmapBuffer, image.imageInfo.rotationDegrees)
        } catch (e: Exception) {
            Log.e(TAG, "Object detection failed", e)
        } finally {
            image.close()
        }
    }

    override fun onResults(
        results: MutableList<Detection>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        if (_fragmentCameraBinding == null) return

        requireActivity().runOnUiThread {
            if (_fragmentCameraBinding == null) return@runOnUiThread
            if (results != null) processDetections(results)
        }
    }

    private fun processDetections(detections: List<Detection>) {
        val best = detections.mapNotNull { detection ->
            val category = detection.categories.firstOrNull()
                ?: return@mapNotNull null
            if (category.score < SPEECH_CONFIDENCE_THRESHOLD) {
                return@mapNotNull null
            }
            category.label.takeIf { it.isNotBlank() }
                ?.let { it to category.score }
        }.maxByOrNull { it.second }

        if (best == null) {
            candidateLabel = ""
            candidateCount = 0
            return
        }

        val label = best.first
        if (label == candidateLabel) {
            candidateCount++
        } else {
            candidateLabel = label
            candidateCount = 1
        }

        if (candidateCount < REQUIRED_STABLE_FRAMES) return

        val now = SystemClock.elapsedRealtime()
        if (now - lastAnnouncementTime < MIN_ANNOUNCEMENT_GAP_MS) return
        if (label == lastAnnouncedLabel &&
            now - lastLabelAnnouncementTime < SAME_LABEL_REPEAT_DELAY_MS
        ) return
        if (speechPending || !ttsReady) return

        lastAnnouncedLabel = label
        lastAnnouncementTime = now
        lastLabelAnnouncementTime = now
        announceObject(label)
    }

    private fun announceObject(label: String) {
        val tts = textToSpeech ?: return
        if (!ttsReady || speechPending) return

        val utteranceId = "object_${++utteranceCounter}"
        val outputFile = File(requireContext().cacheDir, "$utteranceId.wav")
        speechPending = true
        audioFiles[utteranceId] = outputFile

        val synthesisResult = tts.synthesizeToFile(
            "Detected $label",
            Bundle(),
            outputFile,
            utteranceId
        )

        if (synthesisResult == TextToSpeech.ERROR) {
            audioFiles.remove(utteranceId)
            outputFile.delete()
            speechPending = false
            Log.e(TAG, "Unable to synthesize speech")
            return
        }

        val speakResult = tts.speak(
            "Detected $label",
            TextToSpeech.QUEUE_FLUSH,
            null,
            "${utteranceId}_local"
        )
        if (speakResult == TextToSpeech.ERROR) {
            Log.e(TAG, "Unable to speak object label")
        }
    }

    private fun sendAudioToRaspberryPi(file: File) {
        if (!file.exists() || file.length() <= 0L) {
            file.delete()
            return
        }
        if (file.length() > MAX_AUDIO_SIZE) {
            Log.e(TAG, "Audio file exceeds maximum allowed size")
            file.delete()
            return
        }

        if (!::audioExecutor.isInitialized || audioExecutor.isShutdown) {
            file.delete()
            return
        }

        audioExecutor.execute {
            try {
                Socket().use { socket ->
                    socket.connect(
                        InetSocketAddress(PI_HOST, PI_PORT),
                        CONNECT_TIMEOUT_MS
                    )
                    socket.soTimeout = SOCKET_TIMEOUT_MS

                    DataOutputStream(socket.getOutputStream()).use { output ->
                        output.writeInt(file.length().toInt())
                        FileInputStream(file).use { input ->
                            val buffer = ByteArray(8192)
                            var total = 0L
                            while (true) {
                                val count = input.read(buffer)
                                if (count < 0) break
                                total += count
                                if (total > MAX_AUDIO_SIZE) {
                                    throw IllegalStateException("Audio exceeds size limit")
                                }
                                output.write(buffer, 0, count)
                            }
                            output.flush()
                        }
                    }
                }
                Log.d(TAG, "Audio sent to Raspberry Pi")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send audio to Raspberry Pi", e)
            } finally {
                file.delete()
            }
        }
    }

    // This signature matches ObjectDetectorHelper.DetectorListener in the build log.
    override fun onError(error: String) {
        Log.e(TAG, "Detector error: $error")
        showToast(error)
    }

    private fun showToast(message: String) {
        if (!isAdded) return
        requireActivity().runOnUiThread {
            if (isAdded) {
                Toast.makeText(requireContext(), message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
    }

    override fun onDestroyView() {
        try {
            cameraProvider?.unbindAll()
            preview = null
            imageAnalyzer = null
            camera = null
            cameraProvider = null

            if (::cameraExecutor.isInitialized) cameraExecutor.shutdown()
            if (::audioExecutor.isInitialized) audioExecutor.shutdown()

            textToSpeech?.stop()
            textToSpeech?.shutdown()
            textToSpeech = null
            ttsReady = false
            speechPending = false

            audioFiles.values.forEach { it.delete() }
            audioFiles.clear()
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up camera resources", e)
        }

        _fragmentCameraBinding = null
        super.onDestroyView()
    }
}
