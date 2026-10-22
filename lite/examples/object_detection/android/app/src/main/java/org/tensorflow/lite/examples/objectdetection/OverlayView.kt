/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import org.tensorflow.lite.task.vision.detector.Detection

/**
 * Transparent overlay for ThirdEye.
 *
 * Detection results are stored for API compatibility, but this view intentionally
 * draws no bounding boxes, labels, or confidence percentages.
 */
class OverlayView(
    context: Context?,
    attrs: AttributeSet?
) : View(context, attrs) {

    private var results: List<Detection> = emptyList()

    /**
     * Clear stored detection results.
     */
    fun clear() {
        results = emptyList()
        invalidate()
    }

    /**
     * Store detection results without drawing them.
     *
     * CameraFragment can continue processing detections for spoken guidance.
     */
    fun setResults(
        detectionResults: MutableList<Detection>,
        imageHeight: Int,
        imageWidth: Int
    ) {
        results = detectionResults.toList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Intentionally blank: camera preview remains free of visual overlays.
    }
}
