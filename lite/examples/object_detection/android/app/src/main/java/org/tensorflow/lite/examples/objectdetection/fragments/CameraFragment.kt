/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
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
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

class CameraFragment : Fragment(), ObjectDetectorHelper.DetectorListener {

    companion object {
        private const val TAG = "ThirdEye"
        private const val PI_HOST = "10.117.36.42"
        private const val PI_PORT = 5001
        private const val CONNECT_TIMEOUT_MS = 3000
        private const val SOCKET_TIMEOUT_MS = 5000
        private const val MAX_AUDIO_SIZE = 10 * 1024 * 1024
        private const val REQUIRED_STABLE_FRAMES = 2
        private const val MIN_ANNOUNCEMENT_GAP_MS = 700L
        private const val SAME_DESCRIPTION_REPEAT_DELAY_MS = 6000L
        private const val SPEECH_CONFIDENCE_THRESHOLD = 0.60f
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val binding: FragmentCameraBinding
        get() = _fragmentCameraBinding
            ?: throw IllegalStateException("Camera binding is unavailable")

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private var bitmapBuffer: Bitmap? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var audioExecutor: ExecutorService
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    @Volatile private var textToSpeech: TextToSpeech? = null
    @Volatile private var ttsReady = false
    @Volatile private var speechPending = false
    private val audioFiles = ConcurrentHashMap<String, File>()
    private val utteranceCounter = AtomicLong(0L)

    private var candidateDescription = ""
    private var candidateCount = 0
    private var lastAnnouncedDescription = ""
    private var lastAnnouncementTime = 0L
    private var lastDescriptionAnnouncementTime = 0L

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        audioExecutor = Executors.newSingleThreadExecutor()

        binding.viewFinder.implementationMode =
            androidx.camera.view.PreviewView.ImplementationMode.COMPATIBLE

        objectDetectorHelper = ObjectDetectorHelper(
            context = requireContext(),
            objectDetectorListener = this
        )
        initializeTextToSpeech()
        binding.viewFinder.post {
            if (_fragmentCameraBinding != null) setUpCamera()
        }
    }

    private fun initializeTextToSpeech() {
        textToSpeech = TextToSpeech(requireContext()) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val tts = textToSpeech ?: return@TextToSpeech
                val languageResult = tts.setLanguage(Locale.US)
                ttsReady = languageResult != TextToSpeech.LANG_MISSING_DATA &&
                    languageResult != TextToSpeech.LANG_NOT_SUPPORTED

                tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        Log.d(TAG, "TTS started: $utteranceId")
                    }

                    override fun onDone(utteranceId: String?) {
                        if (utteranceId == null) return
                        // The synthesized file has its own ID; only the local speech ID
                        // controls the speechPending flag.
                        if (utteranceId.startsWith("file_")) {
                            audioFiles.remove(utteranceId)?.let(::sendAudioToRaspberryPi)
                        }
                    }

                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        handleTtsError(utteranceId, null)
                    }

                    override fun onError(utteranceId: String?, errorCode: Int) {
                        handleTtsError(utteranceId, errorCode)
                    }
                })
                Log.d(TAG, "Text-to-Speech initialized; ready=$ttsReady")
            } else {
                ttsReady = false
                Log.e(TAG, "Text-to-Speech initialization failed: $status")
            }
        }
    }

    private fun handleTtsError(utteranceId: String?, errorCode: Int?) {
        if (utteranceId == null) return
        if (utteranceId.startsWith("file_")) audioFiles.remove(utteranceId)?.delete()
        Log.e(TAG, "TTS error for $utteranceId; code=$errorCode")
    }

    private fun setUpCamera() {
        val context = context ?: return
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            if (_fragmentCameraBinding == null) return@addListener
            try {
                cameraProvider = providerFuture.get()
                preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setTargetRotation(binding.viewFinder.display.rotation)
                    .build()

                imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setTargetRotation(binding.viewFinder.display.rotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also { analyzer ->
                        analyzer.setAnalyzer(cameraExecutor) { image ->
                            if (_fragmentCameraBinding == null) image.close()
                            else detectObjects(image)
                        }
                    }

                cameraProvider?.unbindAll()
                camera = cameraProvider?.bindToLifecycle(
                    viewLifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
                preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            } catch (e: Exception) {
                Log.e(TAG, "Camera setup failed", e)
                showToast("Unable to start camera: ${e.localizedMessage ?: "unknown error"}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun detectObjects(image: ImageProxy) {
        try {
            if (!::objectDetectorHelper.isInitialized) return
            var bitmap = bitmapBuffer
            if (bitmap == null || bitmap.width != image.width || bitmap.height != image.height) {
                bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                bitmapBuffer = bitmap
            }
            val planeBuffer = image.planes[0].buffer
            planeBuffer.rewind()
            bitmap.copyPixelsFromBuffer(planeBuffer)
            objectDetectorHelper.detect(bitmap, image.imageInfo.rotationDegrees)
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
        val currentBinding = _fragmentCameraBinding ?: return
        activity?.runOnUiThread {
            if (_fragmentCameraBinding !== currentBinding) return@runOnUiThread
            if (results != null) {
                // Keep the standard TensorFlow Lite overlay updated with the detections.
                currentBinding.overlay.setResults(results, imageHeight, imageWidth)
                currentBinding.overlay.invalidate()
                processDetections(results, imageWidth, imageHeight)
            }
        }
    }

    private fun processDetections(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        val best = detections.mapNotNull { detection ->
            val category = detection.categories.firstOrNull() ?: return@mapNotNull null
            if (category.score < SPEECH_CONFIDENCE_THRESHOLD || category.label.isBlank()) {
                return@mapNotNull null
            }
            val box = detection.boundingBox
            val centerX = (box.left + box.right) / 2f
            val boxHeightFraction = (box.height() / max(1, imageHeight)).coerceIn(0f, 1f)
            val direction = when {
                centerX < imageWidth * 0.38f -> "on your left"
                centerX > imageWidth * 0.62f -> "on your right"
                else -> "in front of you"
            }
            // Relative visual estimate only; not a calibrated measurement in metres.
            val distance = when {
                boxHeightFraction >= 0.55f -> "very close"
                boxHeightFraction >= 0.30f -> "nearby"
                boxHeightFraction >= 0.15f -> "at a medium distance"
                else -> "farther away"
            }
            val description = "${category.label}, $direction, appears $distance"
            Triple(detection, category.score, description)
        }.maxByOrNull { it.second }

        if (best == null) {
            candidateDescription = ""
            candidateCount = 0
            return
        }

        val description = best.third
        if (description == candidateDescription) candidateCount++
        else {
            candidateDescription = description
            candidateCount = 1
        }
        if (candidateCount < REQUIRED_STABLE_FRAMES) return

        val now = SystemClock.elapsedRealtime()
        if (now - lastAnnouncementTime < MIN_ANNOUNCEMENT_GAP_MS) return
        if (description == lastAnnouncedDescription &&
            now - lastDescriptionAnnouncementTime < SAME_DESCRIPTION_REPEAT_DELAY_MS
        ) return
        if (speechPending || !ttsReady) return

        lastAnnouncedDescription = description
        lastAnnouncementTime = now
        lastDescriptionAnnouncementTime = now
        announceObject(description)
    }

    private fun announceObject(description: String) {
        val tts = textToSpeech ?: return
        if (!ttsReady || speechPending || _fragmentCameraBinding == null) return
        speechPending = true

        val id = utteranceCounter.incrementAndGet()
        val fileId = "file_$id"
        val outputFile = File(requireContext().cacheDir, "$fileId.wav")
        audioFiles[fileId] = outputFile

        // Synthesize audio only. Do not call tts.speak(), which plays through the phone.
        val text = "Detected $description"
        Log.d(TAG, "Starting WAV synthesis: $text")
        val result = tts.synthesizeToFile(text, Bundle(), outputFile, fileId)
        if (result == TextToSpeech.ERROR) {
            audioFiles.remove(fileId)?.delete()
            speechPending = false
            Log.e(TAG, "Unable to synthesize WAV audio")
        }
    }

    private fun sendAudioToRaspberryPi(file: File) {
        if (!file.exists() || file.length() <= 0L || file.length() > MAX_AUDIO_SIZE) {
            file.delete()
            speechPending = false
            Log.e(TAG, "Audio file missing, empty, or too large")
            return
        }
        if (!::audioExecutor.isInitialized || audioExecutor.isShutdown) {
            file.delete()
            speechPending = false
            return
        }

        try {
            audioExecutor.execute {
                try {
                    val sendStartedAt = SystemClock.elapsedRealtime()
                    Socket().use { socket ->
                        socket.connect(InetSocketAddress(PI_HOST, PI_PORT), CONNECT_TIMEOUT_MS)
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
                                    if (total > MAX_AUDIO_SIZE) error("Audio exceeds size limit")
                                    output.write(buffer, 0, count)
                                }
                                output.flush()
                            }
                        }
                    }
                    Log.d(
                        TAG,
                        "Audio sent to Raspberry Pi at $PI_HOST:$PI_PORT in " +
                            "${SystemClock.elapsedRealtime() - sendStartedAt} ms"
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to send audio to Raspberry Pi", e)
                } finally {
                    file.delete()
                    speechPending = false
                }
            }
        } catch (e: Exception) {
            file.delete()
            speechPending = false
            Log.e(TAG, "Could not queue audio transfer", e)
        }
    }

    override fun onError(error: String) {
        Log.e(TAG, "Detector error: $error")
        showToast(error)
    }

    private fun showToast(message: String) {
        val context = context ?: return
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
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
            bitmapBuffer?.recycle()
            bitmapBuffer = null
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up camera resources", e)
        }
        _fragmentCameraBinding = null
        super.onDestroyView()
    }
}
