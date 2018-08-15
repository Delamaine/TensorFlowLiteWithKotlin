package org.kotlintf.tfkotlinsample

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

/**
 * This class is the recognizer of the image
 *
 * @param interpreter the interpreter for tensorflow lite
 * @param inputSize the resolution of the image for both x-axis and y-axis
 * @param labelList the list of the labels
 *
 */
class Classifier(
        var interpreter: Interpreter? = null,
        var inputSize: Int = 0,
        var labelList: List<String> = emptyList()
)  {
    private val LOG_TAG = "Classifier"
    private val IMAGE_MEAN = 128
    private val IMAGE_STD = 128.0f

    private val MAX_RESULTS = 3
    private val BATCH_SIZE = 1
    private val PIXEL_SIZE = 3
    private val THRESHOLD = 0.0f


    data class Recognition(
            var id: String = "",
            var title: String = "",
            var confidence: Float = 0F
    )  {
        override fun toString(): String {
            return "Title = $title, Confidence = $confidence)"
        }
    }



    companion object {

        /**
         * This method creates the Classifier object as single object (AKA singleton)
         *
         * @throws IOException
         */
        @Throws(IOException::class)
        fun create(assetManager: AssetManager,
                   modelPath: String,
                   labelPath: String,
                   inputSize: Int)
                : Classifier {

            val classifier = Classifier()
            classifier.interpreter = Interpreter(classifier.loadModelFile(assetManager, modelPath))
            classifier.labelList = classifier.loadLabelList(assetManager, labelPath)
            classifier.inputSize = inputSize

            return classifier
        }
    }

    /**
     *  This method loads the model file.
     *
     *  @param assetManager the assetManager
     *  @param modelPath the file path of the model
     *  @return MappedByteBuffer for the model
     */
    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    /**
     * This method loads the label list file
     *
     *  @param assetManager the assetManager
     *  @param labelPath the file path of the label list
     *  @return List of the labels
     */
    @Throws(IOException::class)
    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {
        val labelList = ArrayList<String>()
        val reader = BufferedReader(InputStreamReader(assetManager.open(labelPath)))
        while (true) {
            val line = reader.readLine() ?: break
            labelList.add(line)
        }
        reader.close()
        return labelList
    }






    fun recognizeImage(bitmap: Bitmap): List<Classifier.Recognition> {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
        val result = Array(1) { FloatArray(labelList.size) }
        interpreter!!.run(byteBuffer, result)
        return getSortedResult(result)
    }

    fun close() {
        interpreter!!.close()
        interpreter = null
    }


    /**
     * This method convert bitmap to byte buffer. The assumption for bitmap is that RGB (not ARGB ),
     * so this method ignore from msb to 8th bit. # So PIXEL_SIZE is 3

     * @param bitmap Image Bitmap
     * @return ByteBuffer
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE * 4) // 4はfloat
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputSize * inputSize)
        Log.d(LOG_TAG, "Layout Size:(%d %d) ---- %d".format(bitmap.width, bitmap.height, inputSize * inputSize))

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val `val` = intValues[pixel++]

                byteBuffer.putFloat((((`val`.shr(16)  and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
                byteBuffer.putFloat((((`val`.shr(8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
                byteBuffer.putFloat((((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
            }
        }
        return byteBuffer
    }

    /**
     * This method sorts the result list.
     *
     * @param labelProbArray result form interpreter. The confidences for the label are stored in the coresponding index.
     * @return The sorted list of the Recognitions
     */
    private fun getSortedResult(labelProbArray: Array<FloatArray>): List<Classifier.Recognition> {
        Log.d(LOG_TAG, "List Size:(%d, %d, %d)".format(labelProbArray.size,labelProbArray[0].size,labelList.size))

        val pq = PriorityQueue(
                MAX_RESULTS,
                Comparator<Classifier.Recognition> {
                    (_, _, confidence1), (_, _, confidence2)
                    -> java.lang.Float.compare(confidence1, confidence2) * -1
                })

        for (i in labelList.indices) {
            val confidence = labelProbArray[0][i]
            if (confidence >= THRESHOLD) {
                pq.add(Classifier.Recognition("" + i,
                        if (labelList.size > i) labelList[i] else "Unknown",
                        confidence))
//                Log.d(LOG_TAG, "title: %s,  confidence: %f".format(labelList[i], confidence))

            }
        }
        Log.d(LOG_TAG, "pqsize:(%d)".format(pq.size))

        val recognitions = ArrayList<Classifier.Recognition>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }

        return recognitions
    }
}