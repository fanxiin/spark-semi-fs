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

  def estimateJointEntropy(pairs: Array[(Double, Double)], delta: Double, numBins: Int) = {
//    require((1 / precision % 1) == 0)
    val len = pairs.length
//    val numBins = (1 / precision).toInt
    val bins = Array.ofDim[Int](numBins+1)
    val sortedPairs = pairs.sortBy(_._1)
    var i, j = 0
    for (pair <- sortedPairs) yield {
      val upper = pair._1 + delta
      val lower = pair._1 - delta
      while (j < len && sortedPairs(j)._1 < upper){
        val y = sortedPairs(j)._2
        val lowerOfBin = {
          val tmp = ((y - delta) * numBins).toInt + 1
          if (tmp < 0) 0 else tmp
        }
        val upperOfBin = {
          val tmp = ((y + delta) * numBins).toInt + 1
          if (tmp < numBins) tmp else numBins
        }
        for (k <- lowerOfBin until upperOfBin) bins(k) += 1
        j += 1
      }
      while (sortedPairs(i)._1 < lower){
        val y = sortedPairs(i)._2
        val lowerOfBin =  {
          val tmp = ((y - delta) * numBins).toInt + 1
          if (tmp < 0) 0 else tmp
        }
        val upperOfBin = {
          val tmp = ((y + delta) * numBins).toInt + 1
          if (tmp < numBins) tmp else numBins
        }
        for (k <- lowerOfBin until upperOfBin) bins(k) -= 1
        i += 1
      }
      val approximateBin = math.round(pair._2 * numBins).toInt
//      val approximateBin = (pair._2 * numBins).toInt
      bins(approximateBin)
    }
  }

  def JointEntropy(paris: Array[(Double, Double)], delta: Double) = {
    paris.map(p1 => {
      paris.count(p2 => math.abs(p1._1 - p2._1) < delta && math.abs(p1._2 - p2._2) < delta )
    })
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
//    val a = (1.0D to 20.0D by 1.0).toArray.map(x => (x / 20, 0.0))
    val num = 10000
    val delta = 0.1
    val a = (1 to num).toArray.map(_ => (util.Random.nextDouble(), util.Random.nextDouble())).sortBy(_._1)
//    println(a.mkString(", "))
    val t1 = System.currentTimeMillis()
    val r1 = JointEntropy(a,delta)
    val t2 = System.currentTimeMillis()
    val r2 = estimateJointEntropy(a, delta, 100000)
    val t3 = System.currentTimeMillis()
//    println(r1.sorted.mkString(", "))
//    println(r2.sorted.mkString(", "))
//    val sr1 = r1.sorted
//    val sr2 = r2.sorted
    println(t2-t1)
    println(t3-t2)
//    println(r1.mkString(","))
//    println(r2.mkString(","))
    val diff = r1.zip(r2).filter(a=>a._1 != a._2)
    println(diff.mkString(","))
    println(diff.length)
    println(diff.count(a=>a._2> a._1))
    println(diff.count(a=>a._2< a._1))
//    def locate(x: Double, v:Double, e: Double) ={
//      if (v < x + e && v > x - e) true
//      else false
//    }
//    val t = a.map(p1 => {
//      a.count(p2 => locate(p1._2-p2._2, 0.1, 0.00001))
//    }).count(_ != 0)
//    println(t)
  }
}
