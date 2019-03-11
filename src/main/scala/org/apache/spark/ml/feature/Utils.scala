package org.apache.spark.ml.feature

object Utils {
  def countNeighborHood(values: Array[Double], delta: Double): Array[Int] = {
    val sortedValues = values.sorted
    val len = sortedValues.length
    var i, j, counter = 0
    var upperValue = sortedValues.head - delta
    var lowerValue = sortedValues.head + delta
    val counts = for (k <- 0 until sortedValues.length)
      yield {
        upperValue = sortedValues(k) + delta
        lowerValue = sortedValues(k) - delta
        while (sortedValues(i) < lowerValue){
          i += 1
          counter -= 1
        }
        while (j < len && sortedValues(j) < upperValue){
          j += 1
          counter += 1
        }
        counter
      }
    counts.toArray
  }

  def countNeighborHood1(values: Array[Double], delta: Double): Array[Int] = {
    values.map(v =>{
      values.count(t => Math.abs(v - t)<= delta)
    })
  }

  def log2(x: Double): Double = x match{
    case 0 => 0
    case _ => math.log(x) / math.log(2)
  }

  def estimateMutualInformation(pairs: Array[(Double, Double)], width: Double, numEstimateSteps: Int) = {
    val len = pairs.length
//    require(len > pointsPerVariable)
    val halfWidth = width / 2
    val sortedX = pairs.sortBy(_._1).map(_._1)
    val step = 1.0 / numEstimateSteps
    val estimatePoints = 0D until 1D by step
    val countsX = estimateCounter(sortedX, halfWidth, estimatePoints)
    val sortedByY = pairs.sortBy(_._2)
    val countsY = estimateCounter(sortedByY.map(_._2), halfWidth, estimatePoints)
    val counters = Array.ofDim[Int](estimatePoints.size)
    var i, j = 0
    var upper, lower = 0.0
    val estimateSize = estimatePoints.size
    val tmp = for ((v, countY) <- estimatePoints.zip(countsY)) yield {
      upper = v + halfWidth
      lower = v - halfWidth
      while (j < len && sortedByY(j)._2 < upper) {
        val x = sortedByY(j)._1
        val lowerP = ((x - halfWidth) * numEstimateSteps).asInstanceOf[Int] + 1
        val upperP = ((x + halfWidth) * numEstimateSteps).asInstanceOf[Int]
        for (k <- lowerP to upperP if k < estimateSize) counters(k) += 1
        j += 1
      }
      while (sortedByY(i)._2 < lower) {
        val x = sortedByY(i)._1
        val lowerP = ((x - halfWidth) * numEstimateSteps).asInstanceOf[Int] + 1
        val upperP = ((x + halfWidth) * numEstimateSteps).asInstanceOf[Int]
        for (k <- lowerP to upperP if k < estimateSize) counters(k) -= 1
        i += 1
      }
      counters.zip(countsX).map{
        case (jointCount, countX) =>
          jointCount.asInstanceOf[Double] / len * log2(jointCount * len / (countX * countY))
      }.sum
    }
    -tmp.sum * step * step / (width * width)
  }

  def estimateEntropy(pairs: Array[Double], width: Double, numEstimateSteps: Int) = {
    val len = pairs.length
    //    require(len > pointsPerVariable)
    val halfWidth = width / 2
    val sortedX = pairs.sorted
    val step = 1.0 / numEstimateSteps
    val estimatePoints = 0D until 1D by step
    val countsX = estimateCounter(sortedX, halfWidth, estimatePoints)
    val factor = step / (len * width)
    countsX.map(c => log2(c * factor)).sum * factor * -1
  }

  /**
    * Count the number of sample points that locate in the intervals which are centered on estimatePoints.
    * @param sorted the sample values in ascending order
    * @param halfWidth width of intervals is 2 * halfWidth
    * @param estimatePoint points to estimate
    * @return Array of estimate points counts of each point.
    */
  def estimateCounter(sorted: Array[Double], halfWidth: Double, estimatePoint: Seq[Double]): Array[Int]= {
    val len = sorted.length
    var counter = 0
    var i, j = 0
    var upper, lower = 0.0
    val counts = for (v <- estimatePoint)
      yield {
        upper = v + halfWidth
        lower = v - halfWidth
        while (j < len && sorted(j) < upper) {
          j += 1
          counter += 1
        }
        while (sorted(i) < lower) {
          i += 1
          counter -= 1
        }
        counter
      }
    counts.toArray
  }



  def main(args: Array[String]): Unit = {
//    val result3 = countNeighborHood1(a, 0.15)
//    val result4 = countNeighborHood(a, 0.15)
//    val num = 500
//    val a = (1 to num).toArray.map(_ => scala.util.Random.nextDouble())
//    val t1 = System.currentTimeMillis()
//    val result1 = countNeighborHood1(a, 0.15)
//    val t2 = System.currentTimeMillis()
//    val t3 = System.currentTimeMillis()
//    val result = countNeighborHood(a, 0.15)
//    val t4 = System.currentTimeMillis()
//    println(t2 - t1)
//    println(t4 - t3)
//    println(result1.sorted.mkString(","))
//    println(result.sorted.mkString(","))
    val a = (1.0D to 20.0D by 1.0).toArray.map(x => (x / 20, 0.0))
    println(a.mkString(", "))
    val mi = estimateMutualInformation(a,0.12,10)
    val e1 = estimateEntropy(a.map(_._1),0.12,10)
    val e2 = estimateEntropy(a.map(_._2),0.12,10)
    println(mi +"\t" + e1 + "\t" + e2)
  }
}
