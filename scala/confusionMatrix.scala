import org.apache.spark.mllib.evaluation.MulticlassMetrics

val temp = predictionAndLabels.rdd.map(row => (row.getDouble(0), row.getDouble(1)))
val metrics = new MulticlassMetrics(temp)
val confusionMatrix = metrics.confusionMatrix
