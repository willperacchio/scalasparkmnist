import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{Dataset, DataFrame, SQLContext}
import org.apache.spark.sql.Row
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import org.apache.spark.ml.feature.{VectorAssembler, VectorSizeHint}
import org.apache.spark.ml.linalg.{Vectors, Vector, SparseVector, DenseVector}
import org.apache.spark.sql.types.{StructField, StructType, DoubleType}

def getImageVec(w: Integer, h: Integer, img: String): Vector = {
	val photo = ImageIO.read(new File(img))
	Vectors.dense((0 until w * h).map(x => ((photo.getRGB(x % w, x / w) >> 16 & 0xFF) * 1.0)).to[Array])
}

val path = "/proj/cse398-498/wap221/project/data-full"
val data = new Array[(Double, Vector)](100 * 10 * 15)

var count = 0
for (p <- 1 to 100) {
	for (t <- 1 to 10) {
		for (n <- 1 to 15) {
			val file = "Locate{" + p + "," + t + "," + n + "}.jpg"
			// print(file + "\n")
			data(count) = (n, getImageVec(64, 64, path + "/" + file).toSparse)
			count += 1
		}
	}
}

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

val schema = StructType(Seq(
	StructField("label", DoubleType, false),
	StructField("features", VectorType, false)
))

val df_temp = spark.createDataFrame(data).toDF("label", "_2")
val sizeHint = new VectorSizeHint().setInputCol("_2").setSize(4096)
val df_withSize = sizeHint.transform(df_temp)
val assembler = new VectorAssembler().setInputCols(Array("_2")).setOutputCol("features").setHandleInvalid("keep")
val df = assembler.transform(df_withSize)

val testPercentage = 0.2
val seed = 498L
val Array(train, test) = df.randomSplit(Array(1 - testPercentage, testPercentage), seed)
