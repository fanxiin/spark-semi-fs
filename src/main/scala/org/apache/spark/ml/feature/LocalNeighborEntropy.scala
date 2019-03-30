package org.apache.spark.ml.feature

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}

import scala.collection.mutable.ArrayBuffer

trait LocalNeighborEntropy {
  protected def log2(x: Double): Double = x match {
    case 0 => 0
    case _ => math.log(x) / math.log(2)
  }

  /**
    * Fast counting the neighbor counts of elements in values. The ordering of elements not guaranteed.
    * @param values Array of values of one attributes.
    * @param delta  Threshold of neighbor relationship.
    * @return Numbers of elements' neighbor
    */
  protected def neighborCounts(
      values: Array[Double],
      scaleDelta: Double): Array[Int] = {
    val sortedValues = values.sorted
    val len = sortedValues.length
    var i, j, counter = 0
    var upperValue = sortedValues.head - scaleDelta
    var lowerValue = sortedValues.head + scaleDelta
    val counts = for (v <- sortedValues)
      yield {
        upperValue = v + scaleDelta
        lowerValue = v - scaleDelta
        while (sortedValues(i) < lowerValue) {
          i += 1
          counter -= 1
        }
        while (j < len && sortedValues(j) <= upperValue) {
          j += 1
          counter += 1
        }
        counter
      }
    counts
  }


  def entropy(col: ColData, delta: Double): Double

 // def jointEntropy(col1: ColData, col2: ColData, delta: Double): Double

  def jointEntropy(col1: ColData, col2: ColData, delta: Double, numBins: Int = 10000): Double
}

object LocalNeighborEntropy {
  def entropy(col: ColData, delta: Double): Double = col.vector match {
    case _: SparseVector =>
      LocalSparseNeighborEntropy.entropy(col, delta)
    case _: DenseVector =>
      LocalDensNeighborEntropy.entropy(col,delta)
  }

  def jointEntropy(
      col1: ColData,
      col2: ColData,
      delta: Double,
      numBins: Int = 10000): Double = (col1.vector,col2.vector) match {
    case (_: SparseVector, _: SparseVector) =>
      LocalSparseNeighborEntropy.jointEntropy(col1,col2,delta,numBins)
    case (_: DenseVector, _: DenseVector) =>
      LocalDensNeighborEntropy.jointEntropy(col1,col2,delta,numBins)
//    case (_: SparseVector, _: DenseVector) =>
//      LocalDensNeighborEntropy.jointEntropy(col2,col1,delta,numBins)
//    case (_: DenseVector, _: SparseVector) =>
//      LocalDensNeighborEntropy.jointEntropy(col1,col2,delta,numBins)

  }

}

private object LocalDensNeighborEntropy extends LocalNeighborEntropy {
  def zipDens(c1: ColData, c2: ColData): Array[(Double, Double)] =
    c1.vector.asInstanceOf[DenseVector].values.zip(c2.vector.asInstanceOf[DenseVector].values)


  def countNeighborCountOrigin(values: Array[Double], delta: Double): Array[Int] = {
    values.map(v => {
      values.count(t => Math.abs(v - t) <= delta)
    })
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
    case NumericalColData(_, v: DenseVector, max, min) => entropy(v.values, delta*(max-min))
    case NominalColData(_, v: DenseVector) => entropy(v.values)
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
  def pureNumJointEntropy(col1: NumericalColData, col2: NumericalColData, delta: Double, numBins: Int = 100000):Double = {
    val size = col1.vector.size.toDouble
    val counts = estimateNeighborCounts(col1, col2, delta, numBins)
    counts.map(c => log2(c / size)).sum / size * -1
  }


  /**
    * Compute the delta-neighbor joint entropy between nominal and numerical variable. Use infinite norm as the
    * measure of neighborhood relationship.The ordering of elements not guaranteed.
    *
    * @param data    2-dimension data. The first value is nominal value.
    * @param delta   The threshold of neighborhood relationship.
    * @return neighbor entropy.
    */
  def mixJointEntropy(col1: NominalColData, col2: NumericalColData, delta: Double): Double = {
    val scaledDelta = delta * (col2.max - col2.min)
    val size = col1.vector.size.toDouble
    val data = zipDens(col1, col2)
    val groupedData = data.groupBy(_._1).map(_._2.unzip._2)
    groupedData.flatMap(neighborCounts(_, scaledDelta)).map(c => log2( c / size)).sum / size * -1
  }

  /**
    * Compute the (delta-neighbor) joint entropy of two nominal variable.
    *
    * @param data    2-dimension data.
    * @return (neighbor) entropy.
    */
  def pureNomJointEntropy(data: Array[(Double, Double)]): Double = {
    val len = data.length.toDouble
    val (maxX, maxY) = data.reduce[(Double,Double)]{
      case (v, that) => (math.max(v._1, that._1), math.max(v._2, that._2))
    }
    val contingency = Array.ofDim[Int](maxX.toInt + 1, maxY.toInt + 1)
    for ((x, y) <- data) contingency(x.toInt)(y.toInt) += 1
    contingency.flatten.map(c => c * log2(c / len)).sum / len * -1
  }

  def jointEntropy(col1: ColData, col2: ColData, delta: Double, numBins: Int = 10000): Double = {
    (col1,col2) match {
      case (c1: NominalColData, c2: NominalColData) =>
        pureNomJointEntropy(zipDens(c1,c2))
      case (c1: NominalColData, c2: NumericalColData) =>
        mixJointEntropy(c1,c2, delta)
      case (c1: NumericalColData, c2: NominalColData) =>
        mixJointEntropy(c2,c1, delta)
      case (c1: NumericalColData, c2: NumericalColData) =>
        pureNumJointEntropy(c1, c2, delta, numBins)
    }
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
  private def estimateNeighborCounts(col1: NumericalColData, col2: NumericalColData, delta: Double, numBins: Int = 100000): Array[Int] = {
    //    require((1 / precision % 1) == 0)
    val data = col1.vector.asInstanceOf[DenseVector].values.zip(col2.vector.asInstanceOf[DenseVector].values)
    val scaledDelta1 = delta * (col1.max - col1.min)
    val scaledDelta2 = delta * (col2.max - col2.min)
    val lenY = col2.max - col2.min
    val minY = col2.min
    val len = data.length
    val bins = Array.ofDim[Int](numBins + 1)

    val scalar = numBins / lenY
    def lowerUpper(relativeValue: Double): (Int, Int) = {
      val lowerOfBin = {
        val tmp = math.ceil((relativeValue - scaledDelta2) * scalar).toInt
        if (tmp < 0) 0 else tmp
      }
      val upperOfBin = {
        val tmp = math.ceil((relativeValue + scaledDelta2) * scalar).toInt
        if (tmp < numBins) tmp else numBins
      }
      (lowerOfBin, upperOfBin)
    }

    val sortedPairs = data.sortBy(_._1)
    var i, j = 0
    for (pair <- sortedPairs) yield {
      val upper = pair._1 + scaledDelta1
      val lower = pair._1 - scaledDelta1
      while (j < len && sortedPairs(j)._1 <= upper) {
        val relativeY = sortedPairs(j)._2 - minY
        val (lowerOfBin, upperOfBin) = lowerUpper(relativeY)
        // Because the last index of bin is numBins(length of bins is numBins+1).
        for (k <- lowerOfBin to upperOfBin) bins(k) += 1
        j += 1
      }
      while (sortedPairs(i)._1 < lower) {
        val relativeY = sortedPairs(i)._2 - minY
        val (lowerOfBin, upperOfBin) = lowerUpper(relativeY)
        for (k <- lowerOfBin to upperOfBin) bins(k) -= 1
        i += 1
      }
      val approximateBin = math.round((pair._2 - minY) / lenY * numBins).toInt
      bins(approximateBin)
    }
  }

  def exactNeighborCounts(paris: Array[(Double, Double)], delta: Double): Array[Int] = {
    paris.map(p1 => {
      paris.count(p2 => math.abs(p1._1 - p2._1) <= delta && math.abs(p1._2 - p2._2) <= delta)
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
      val count_xy = paris.count(p2 => math.abs(p1._1 - p2._1) <= delta && math.abs(p1._2 - p2._2) <= delta)
      val count_x = paris.count(p2 => math.abs(p1._1 - p2._1) <= delta)
      val count_y = paris.count(p2 => math.abs(p1._2 - p2._2) <= delta)
      log2(count_x * count_y / (len * count_xy))
    }).sum / len * -1
  }
}

private object LocalSparseNeighborEntropy extends LocalNeighborEntropy {

//  def zipSparseCol(col1: ColData, col2: ColData): Array[(Double, Double)] = {
//    val vector1 = col1.vector.asInstanceOf[SparseVector]
//    val vector2 = col2.vector.asInstanceOf[SparseVector]
//    // breeze SparseArray is faster than map implement
//    def sparseArray(v: SparseVector): SparseArray[Double] =
//      new SparseArray[Double](v.indices, v.values, v.indices.length, v.size, 0.0)
//    val sArray1 = sparseArray(vector1)
//    val sArray2 = sparseArray(vector2)
//    val zippedIndices = (vector1.indices ++ vector2.indices).distinct
//    zippedIndices.map(index=> (sArray1(index), sArray2(index)))
//  }

  def zipSparseCol(col1: ColData, col2: ColData): Array[(Double, Double)] = {
    val vector1 = col1.vector.asInstanceOf[SparseVector]
    val vector2 = col2.vector.asInstanceOf[SparseVector]
    val (indices1,values1) = (vector1.indices, vector1.values)
    val (indices2,values2) = (vector2.indices, vector2.values)
    var (i, j) = (0,0)
    val dataBuffer = new ArrayBuffer[(Double, Double)](math.max(indices1.length, indices2.length))
    while (i < indices1.length && j < indices2.length) {
      if (indices1(i) == indices2(j)){
        dataBuffer += ((values1(i), values2(j)))
        i += 1
        j += 1
      }
      else if (indices1(i) < indices2(j)){
        dataBuffer += ((values1(i), 0.0))
        i += 1
      }
      else if (indices1(i) > indices2(j)){
        dataBuffer += ((0.0, values2(j)))
        j += 1
      }
    }
    for (k <- i until indices1.length) dataBuffer += ((values1(k), 0.0))
    for (k <- j until indices2.length) dataBuffer += ((0.0, values2(k)))
    dataBuffer.toArray
  }

//  def zipSparseCol(col1: ColData, col2: ColData): Array[(Double, Double)] = {
//    val vector1 = col1.vector.asInstanceOf[SparseVector]
//    val vector2 = col2.vector.asInstanceOf[SparseVector]
//    val map1 = vector1.indices.zip(vector1.values).toMap
//    val map2 = vector2.indices.zip(vector2.values).toMap
//    val zippedIndices = (vector1.indices ++ vector2.indices).distinct
//    zippedIndices.map(index=> (map1.getOrElse(index,0.0), map2.getOrElse(index, 0.0)))
//  }

  private def sparseNeighborCounts(colData: NumericalColData, delta: Double): (Array[Int],(Int,Int)) = {
    val scaledDelta = delta * (colData.max - colData.min)
    val vector = colData.vector.asInstanceOf[SparseVector]
    val values = vector.values
    val numZeros = vector.size - values.length
    val sortedValues = values.sorted
    val len = sortedValues.length
    var i, j, counter = 0
    var upperValue = 0.0
    var lowerValue = 0.0
    var zeroNeighborCount = numZeros

//    if (sortedValues.length == 0) return (Array.empty[Int],(numZeros, zeroNeighborCount))

    val counts = for (v <- sortedValues)
      yield {
        upperValue = v + scaledDelta
        lowerValue = v - scaledDelta
        while (sortedValues(i) < lowerValue) {
          i += 1
          counter -= 1
        }
        while (j < len && sortedValues(j) <= upperValue) {
          j += 1
          counter += 1
        }
        if (math.abs(v) <= scaledDelta){
          zeroNeighborCount += 1
          counter + numZeros
        }
        else
          counter
      }
    (counts, (numZeros, zeroNeighborCount))
  }

  private def sparseEstimateNeighborCounts(
      col1: NumericalColData,
      col2: NumericalColData,
      delta: Double,
      numBins: Int): (Array[Int],(Int,Int))= {
    //    require((1 / precision % 1) == 0)
    val len1 = col1.max - col1.min
    val len2 = col2.max - col2.min
    val minY = col2.min
    val (scaledDelta1, scaledDelta2)  = (len1 * delta,len2 * delta)
    val data = zipSparseCol(col1,col2)
    val numZeros = col1.vector.size - data.length
    var zeroNeighborCount = numZeros
    val len = data.length
    val bins = Array.ofDim[Int](numBins + 1)

    val scalar = numBins / len2
    def lowerUpper(relativeValue: Double): (Int, Int) = {
      val lowerOfBin = {
        val tmp = math.ceil((relativeValue - scaledDelta2) * scalar).toInt
        if (tmp < 0) 0 else tmp
      }
      val upperOfBin = {
        val tmp = math.ceil((relativeValue + scaledDelta2) * scalar).toInt
        if (tmp < numBins) tmp else numBins
      }
      (lowerOfBin, upperOfBin)
    }

    val sortedPairs = data.sortBy(_._1)
    var i, j = 0
    val counts = for (pair <- sortedPairs) yield {
      val upper = pair._1 + scaledDelta1
      val lower = pair._1 - scaledDelta1
      while (j < len && sortedPairs(j)._1 <= upper) {
        val relativeY = sortedPairs(j)._2 - minY
        val (lowerOfBin, upperOfBin) = lowerUpper(relativeY)
        // Because the last index of bin is numBins(length of bins is numBins+1).
        for (k <- lowerOfBin to upperOfBin) bins(k) += 1
        j += 1
      }
      while (sortedPairs(i)._1 < lower) {
        val relativeY = sortedPairs(i)._2 - minY
        val (lowerOfBin, upperOfBin) = lowerUpper(relativeY)
        for (k <- lowerOfBin to upperOfBin) bins(k) -= 1
        i += 1
      }
      val approximateBin = math.round((pair._2 - minY) / len2 * numBins).toInt
      if (math.abs(pair._1) <= scaledDelta1 && math.abs(pair._2) <= scaledDelta2){
        zeroNeighborCount += 1
        bins(approximateBin) + numZeros
      } else
        bins(approximateBin)
    }
    (counts, (numZeros, zeroNeighborCount))
  }

  def sparseNumericalEntropy(col: NumericalColData, delta: Double):Double = {
    val len = col.vector.size.toDouble
    val (counts, (numZero, zeroNeighborCount)) = sparseNeighborCounts(col, delta)
    (counts.map(c => log2(c / len)).sum + log2(zeroNeighborCount / len) * numZero) / len * -1
  }

  def sparseNominalEntropy(col: NominalColData):Double = {
    val size = col.vector.size.toDouble
    val data = col.vector.asInstanceOf[SparseVector].values
    val numZeros = size - data.length.toDouble
    val maxX = data.max.toInt
    val table = Array.ofDim[Int](maxX + 1)
    data.foreach(v=> table(v.toInt) += 1)
    (table.map(c => c * log2(c / size)).sum + numZeros * log2(numZeros/size)) / size * -1
  }

  def sparsePureNomJointEntropy(col1: NominalColData, col2: NominalColData): Double = {
    val size = col1.vector.size.toDouble
    val data = zipSparseCol(col1, col2)
    val numZeros = size - data.length
    val (maxX, maxY) = data.reduce[(Double,Double)]{
      case (v, that) => (math.max(v._1, that._1), math.max(v._2, that._2))
    }
    val contingency = Array.ofDim[Int](maxX.toInt + 1, maxY.toInt + 1)
    for ((x, y) <- data) contingency(x.toInt)(y.toInt) += 1
    (contingency.flatten.map(c => c * log2(c / size)).sum + numZeros * log2(numZeros / size)) /size * -1
  }


  def sparsePureNumJointEntropy(col1: NumericalColData, col2: NumericalColData, delta: Double, numBins: Int = 100000):Double = {
    val len = col1.vector.size.toDouble
    val (counts, (numZero, zeroNeighborCount)) = sparseEstimateNeighborCounts(col1, col2, delta, numBins)
    (counts.map(c => log2(c / len)).sum + log2(zeroNeighborCount / len) * numZero) / len * -1
  }

  /**
    * Compute the delta-neighbor joint entropy between nominal and numerical variable. Use infinite norm as the
    * measure of neighborhood relationship.The ordering of elements not guaranteed.
    *
    * @param data    2-dimension data. The first value is nominal value.
    * @param delta   The threshold of neighborhood relationship.
    * @return neighbor entropy.
    */
  def sparseMixJointEntropy(col1: NominalColData, col2: NumericalColData, delta: Double): Double = {
    val scaledDelta = (col2.max - col2.min) * delta
    val size = col1.vector.size.toDouble

    val data = zipSparseCol(col1, col2)
    val numZeros = col1.vector.size - data.length
    val groupedData = data.groupBy(_._1)
    val firstNoneZeroData = (groupedData - 0).map(_._2.unzip._2)
    val firstZeroData = groupedData(0).map(_._2).sorted

    var zeroNeighborCount = numZeros
    val firstZeroDataCounts = neighborCounts(firstZeroData, scaledDelta)
    for (i <- firstZeroData.indices) {
      if (math.abs(firstZeroData(i)) <= scaledDelta){
        zeroNeighborCount += 1
        firstZeroDataCounts(i) += numZeros
      }
    }
    val noneZeroPart = firstNoneZeroData.flatMap(neighborCounts(_, scaledDelta)).map(c => log2( c / size)).sum
    val zeroPart = firstZeroDataCounts.map(c => log2( c / size)).sum + numZeros * log2(zeroNeighborCount / size)
    (noneZeroPart + zeroPart) / size * -1
  }

  override def entropy(col: ColData, delta: Double): Double = col match {
    case c: NumericalColData => sparseNumericalEntropy(c, delta)
    case c: NominalColData => sparseNominalEntropy(c)
  }

  override def jointEntropy(col1: ColData, col2: ColData, delta: Double, numBins: Int): Double = (col1,col2) match {
    case (c1: NominalColData, c2: NominalColData) =>
      sparsePureNomJointEntropy(c1, c2)
    case (c1: NominalColData, c2: NumericalColData) =>
      sparseMixJointEntropy(c1,c2, delta)
    case (c1: NumericalColData, c2: NominalColData) =>
      sparseMixJointEntropy(c2,c1, delta)
    case (c1: NumericalColData, c2: NumericalColData) =>
      sparsePureNumJointEntropy(c1,c2, delta, numBins)
  }
}