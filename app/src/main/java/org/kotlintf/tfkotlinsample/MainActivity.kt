package org.kotlintf.tfkotlinsample

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.app.Activity
import java.io.FileNotFoundException
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {
    private val CHOOSE_IMAGE = 1001
    private val MODEL_PATH = "optimized_graph.tflite"
    private val LABEL_PATH = "retrained_labels.txt"
    private val INPUT_SIZE = 224

    private lateinit var photoImage: Bitmap
    lateinit var classifier: Classifier


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        try {
            classifier = Classifier.create(
                    assets,
                    MODEL_PATH,
                    LABEL_PATH,
                    INPUT_SIZE)
        } catch (e: Exception) {
            throw RuntimeException("Error initializing TensorFlow!", e)
        }
        imageView.setOnClickListener(){
            choosePicture()
        }
    }
    private fun choosePicture() {
        val intent = Intent()
        intent.type = "image/*"
        intent.action = Intent.ACTION_GET_CONTENT
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        startActivityForResult(intent, CHOOSE_IMAGE)
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == CHOOSE_IMAGE && resultCode == Activity.RESULT_OK)
            try {
                val stream = contentResolver!!.openInputStream(data!!.getData())
                if (::photoImage.isInitialized) photoImage.recycle()
                photoImage = BitmapFactory.decodeStream(stream)
                photoImage = Bitmap.createScaledBitmap(photoImage, INPUT_SIZE, INPUT_SIZE, false)
                imageView.setImageBitmap(photoImage)
                var list=classifier.recognizeImage(photoImage)
                var f=list.firstOrNull()
                textView.text=f?.title+"::::"+f?.confidence

            } catch (e: FileNotFoundException) {
                e.printStackTrace()
            }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }

}
