# spark-semi-fs
semi-supervised feature selection algorithm for Spark 

# Usage
```scala
    val selector = new SemiSelector()
      .setNumTopFeatures(50)
      .setDelta(0.1)
      .setNumPartitions(10)
      .isSparse(true)
    val model = selector.fit(data)
    val result = model.transform(data)
```
