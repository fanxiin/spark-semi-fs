package org.apache.spark.ml.feature

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}

object LocalNeighborEntropy {

  /**
    * Fast counting the neighbor counts of elements in values. The ordering of elements not guaranteed.
    * @param values Array of values of one attributes.
    * @param delta  Threshold of neighbor relationship.
    * @return Numbers of elements' neighbor
    */
  private def neighborCounts(values: Array[Double], delta: Double): Array[Int] = {
    val sortedValues = values.sorted
    val len = sortedValues.length
    var i, j, counter = 0
    var upperValue = sortedValues.head - delta
    var lowerValue = sortedValues.head + delta
    val counts = for (v <- sortedValues)
      yield {
        upperValue = v + delta
        lowerValue = v - delta
        while (sortedValues(i) < lowerValue) {
          i += 1
          counter -= 1
        }
        while (j < len && sortedValues(j) < upperValue) {
          j += 1
          counter += 1
        }
        counter
      }
    counts
  }

  def countNeighborCountOrigin(values: Array[Double], delta: Double): Array[Int] = {
    values.map(v => {
      values.count(t => Math.abs(v - t) < delta)
    })
  }

  private def log2(x: Double): Double = x match {
    case 0 => 0
    case _ => math.log(x) / math.log(2)
  }

  /**
    * Compute single numerical variable neighbor entropy.
    *
    * @param data  1-dimension data.
    * @param delta The threshold of neighborhood relationship.
    * @return neighbor entropy
    */
  def entropy(data: Array[Double], delta: Double): Double = {
    val len = data.length.toDouble
    val counts = neighborCounts(data, delta)
    counts.map(c => log2(c / len)).sum / len * -1
  }

  /**
    * Compute single nominal variable (neighbor) entropy.
    *
    * @param data  1-dimension discrete data.
    * @return neighbor entropy
    */
  def entropy(data: Array[Double]): Double = {
    val len = data.length.toDouble
    val maxX = data.max.toInt
    val table = Array.ofDim[Int](maxX + 1)
    data.foreach(v=> table(v.toInt) += 1)
    table.map(c => c * log2(c / len)).sum / len * -1
  }

  def entropy(colData: ColData, delta: Double): Double = colData match {
    case NumericalColData(_, v: DenseVector) => entropy(v.values, delta)
    case NumericalColData(_, v: SparseVector) =>
      entropy(v.values, delta)
    case NominalColData(_, v: DenseVector) => entropy(v.values)
    case NominalColData(_, v: SparseVector) =>
      entropy(v.values)
  }

  /**
    * Estimate the delta-neighbor joint entropy between two numerical variable. Use infinite norm as the measure of
    * neighborhood relationship.The ordering of elements not guaranteed.
    *
    * @param data    2-dimension data.
    * @param delta   The threshold of neighborhood relationship.
    * @param numBins Number of bins to estimate the count.
    * @return estimate of neighbor entropy.
    */
  def pureNumJointEntropy(data: Array[(Double, Double)], delta: Double, numBins: Int = 100000): Double = {
    val len = data.length.toDouble
    val counts = estimateNeighborCounts(data, delta, numBins)
    counts.map(c => log2(c / len)).sum / len * -1
  }

  /**
    * Compute the delta-neighbor joint entropy between nominal and numerical variable. Use infinite norm as the
    * measure of neighborhood relationship.The ordering of elements not guaranteed.
    *
    * @param data    2-dimension data. The first value is nominal value.
    * @param delta   The threshold of neighborhood relationship.
    * @return neighbor entropy.
    */
  def mixJointEntropy(data: Array[(Double, Double)], delta: Double): Double = {
    val len = data.length.toDouble
    val groupedData = data.groupBy(_._1).map(_._2.unzip._2)
    groupedData.flatMap(neighborCounts(_, delta)).map(c => log2( c / len)).sum / len * -1
  }

  /**
    * Compute the (delta-neighbor) joint entropy of two nominal variable.
    *
    * @param data    2-dimension data.
    * @return (neighbor) entropy.
    */
  def pureNomJointEntropy(data: Array[(Double, Double)]): Double = {
    val len = data.length.toDouble
    def max(a: Double, b: Double) = if (a > b) a else b
    val (maxX, maxY) = data.reduce[(Double,Double)]{
      case (v, that) => (max(v._1, that._1), max(v._2, that._2))
    }
    val contingency = Array.ofDim[Int](maxX.toInt + 1, maxY.toInt + 1)
    for ((x, y) <- data) contingency(x.toInt)(y.toInt) += 1
    contingency.flatten.map(c => c * log2(c / len)).sum / len * -1
  }

  def jointEntropy(col1: ColData, col2: ColData, delta: Double): Double = (col1,col2) match {
    case (c1: NominalColData, c2: NominalColData) =>
      pureNomJointEntropy(c1.vector.toDense.values.zip(c2.vector.toDense.values))
    case (c1: NominalColData, c2: NumericalColData) =>
      mixJointEntropy(c1.vector.toDense.values.zip(c2.vector.toDense.values), delta)
    case (c1: NumericalColData, c2: NominalColData) =>
      mixJointEntropy(c2.vector.toDense.values.zip(c1.vector.toDense.values), delta)
    case (c1: NumericalColData, c2: NumericalColData) =>
      pureNumJointEntropy(c1.vector.toDense.values.zip(c2.vector.toDense.values), delta)
  }

  /**
    * Estimate the delta-neighbor counts of pairs. Values of column must between 0 and 1! Use infinite norm as the
    * measure of neighborhood relationship. The ordering of elements not guaranteed.
    *
    * @param data    2-dimension data.
    * @param delta   The threshold of neighborhood relationship.
    * @param numBins Number of bins to estimate the count.
    * @return Approximate numbers of elements' neighbor.
    */
  private def estimateNeighborCounts(data: Array[(Double, Double)], delta: Double, numBins: Int): Array[Int] = {
    //    require((1 / precision % 1) == 0)
    val len = data.length
    val bins = Array.ofDim[Int](numBins + 1)
    val sortedPairs = data.sortBy(_._1)
    var i, j = 0
    for (pair <- sortedPairs) yield {
      val upper = pair._1 + delta
      val lower = pair._1 - delta
      while (j < len && sortedPairs(j)._1 < upper) {
        val y = sortedPairs(j)._2
        val lowerOfBin = {
          val tmp = ((y - delta) * numBins).toInt + 1
          if (tmp < 0) 0 else tmp
        }
        val upperOfBin = {
          val tmp = ((y + delta) * numBins).toInt + 1
          if (tmp < numBins) tmp else numBins
        }
        // Because the last index of bin is numBins(length of bins is numBins+1).
        for (k <- lowerOfBin to upperOfBin) bins(k) += 1
        j += 1
      }
      while (sortedPairs(i)._1 < lower) {
        val y = sortedPairs(i)._2
        val lowerOfBin = {
          val tmp = ((y - delta) * numBins).toInt + 1
          if (tmp < 0) 0 else tmp
        }
        val upperOfBin = {
          val tmp = ((y + delta) * numBins).toInt + 1
          if (tmp < numBins) tmp else numBins
        }
        for (k <- lowerOfBin to upperOfBin) bins(k) -= 1
        i += 1
      }
      val approximateBin = math.round(pair._2 * numBins).toInt
      bins(approximateBin)
    }
  }

  def exactNeighborCounts(paris: Array[(Double, Double)], delta: Double): Array[Int] = {
    paris.map(p1 => {
      paris.count(p2 => math.abs(p1._1 - p2._1) < delta && math.abs(p1._2 - p2._2) < delta)
    })
  }

  def exactNJE(data: Array[(Double, Double)], delta: Double): Double = {
    val len = data.length.toDouble
    val counts = exactNeighborCounts(data, delta)
    counts.map(c => log2(c / len)).sum / len * -1
  }


  def exactNMI(paris: Array[(Double, Double)], delta: Double): Double = {
    val len = paris.length.toDouble
    paris.map(p1 => {
      val count_xy = paris.count(p2 => math.abs(p1._1 - p2._1) < delta && math.abs(p1._2 - p2._2) < delta)
      val count_x = paris.count(p2 => math.abs(p1._1 - p2._1) < delta)
      val count_y = paris.count(p2 => math.abs(p1._2 - p2._2) < delta)
      log2(count_x * count_y / (len * count_xy))
    }).sum / len * -1
  }
}

