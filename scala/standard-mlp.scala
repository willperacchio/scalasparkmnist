val layers2 = Array(784, 32, 16, 10)
val mlpSeed2 = 498
val tol2 = 1E-4
val blockSize2 = 128
val maxIter2 = 100

val trainer2 = new MultilayerPerceptronClassifier().setLayers(layers2)
			.setTol(tol2).setBlockSize(blockSize2).setSeed(mlpSeed2).setMaxIter(maxIter2)

val model2 = trainer2.fit(train2)
val result2 = model2.transform(test2)
