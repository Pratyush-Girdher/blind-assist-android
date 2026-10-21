        }

        val binding = _fragmentCameraBinding ?: return

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setBackpressureStrategy(
                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
            )
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { analyzer ->
                analyzer.setAnalyzer(cameraExecutor) { image ->
                    try {
                        if (_fragmentCameraBinding == null) {
                            return@setAnalyzer
                        }

                        if (
                            !::bitmapBuffer.isInitialized ||
                            bitmapBuffer.width != image.width ||
                            bitmapBuffer.height != image.height
                        ) {
                            bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,
                                Bitmap.Config.ARGB_8888
                            )
                        }

                        detectObjects(image)
                    } catch (e: Exception) {
                        Log.e(TAG, "Object detection failed", e)
                    } finally {
                        image.close()
                    }
                }
            }

        provider.unbindAll()

        try {
            camera = provider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
            showToast("Unable to start the camera preview.")
        }
    }

    private fun detectObjects(image: ImageProxy) {
        if (!::bitmapBuffer.isInitialized) return

        bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)

        objectDetectorHelper.detect(
            bitmapBuffer,
            image.imageInfo.rotationDegrees
        )
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)

        _fragmentCameraBinding?.let { binding ->
            imageAnalyzer?.targetRotation =
                binding.viewFinder.display.rotation
        }
    }

    override fun onResults(
        results: MutableList<Detection>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        activity?.runOnUiThread {
            val binding = _fragmentCameraBinding
                ?: return@runOnUiThread

            binding.overlay.setResults(
                results ?: LinkedList(),
                imageHeight,
                imageWidth
            )

            binding.overlay.invalidate()
            announceDetection(results)
        }
    }

    override fun onError(error: String) {
        showToast(error)
    }

    private fun showToast(message: String) {
        activity?.runOnUiThread {
            context?.let { safeContext ->
                Toast.makeText(
                    safeContext,
                    message,
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }
}
