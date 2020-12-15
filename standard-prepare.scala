import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row


val path = "/proj/cse398-498/course/MLP/mnist"
val df = spark.read.format("libsvm").option("inferSchema", "true").load(path)

df.show(1, false)

val testPercentage = 0.2
val seed = 498L
val Array(train, test) = df.randomSplit(Array(1 - testPercentage, testPercentage), seed)

train.head()
train.count()
test.head()
test.count()
