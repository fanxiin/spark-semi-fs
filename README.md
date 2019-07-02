# spark-semi-fs
semi-supervised feature selection algorithm for Spark 

# Useage
```scala
    val selector = new SemiSelector()
      .setNumTopFeatures(32)
      .setDelta(0.1)
      .setNumPartitions(10)
      .isSparse(true)
      .setNominalIndices(Array(0))
    val model = selector.fit(data)
    val result = model.transform(data)
```
