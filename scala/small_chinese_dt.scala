// import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
// import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
// import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}

// import org.apache.spark.ml.linalg.{Vectors, Vector}

val trees = 50
val mlpSeed = 498
val tol = 1E-4
val blockSize = 128
val maxIter = 100

// val train = spark.read.parquet("train.parquet")
// val test = spark.read.parquet("test.parquet")
// train.cache()
// test.cache()

val trainer = new DecisionTreeClassifier()

val model = trainer.fit(train)
val result = model.transform(test)


val predictionAndLabels = result.select("prediction", "label")

val acc = new MulticlassClassificationEvaluator().setMetricName("accuracy")
val prec = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision")
val recall = new MulticlassClassificationEvaluator().setMetricName("weightedRecall")
val f1 = new MulticlassClassificationEvaluator().setMetricName("f1")

println("Accuracy  =\t" + acc.evaluate(predictionAndLabels))
println("Precision =\t" + prec.evaluate(predictionAndLabels))
println("Recall    =\t" + recall.evaluate(predictionAndLabels))
println("F1 Score  =\t" + f1.evaluate(predictionAndLabels))

