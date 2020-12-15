import org.apache.spark.sql.Row
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

val layers = Array(780, 900, 10)
val mlpSeed = 498
val tol = 1E-4
val blockSize = 128
val maxIter = 100

val trainer = new MultilayerPerceptronClassifier().setLayers(layers)
			.setTol(tol).setBlockSize(blockSize).setSeed(mlpSeed).setMaxIter(maxIter)

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

