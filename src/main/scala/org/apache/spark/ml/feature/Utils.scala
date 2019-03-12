package org.apache.spark.ml.feature

object Utils {
  /**
    * Fast counting the neighbor counts of elements in values. The ordering of elements not guaranteed.
    * @param values Array of values of one attributes.
    * @param delta  Threshold of neighbor relationship.
    * @return Numbers of elements' neighbor
    */
  def neighborCounts(values: Array[Double], delta: Double): Array[Int] = ???

  def countNeighborCountOrigin(values: Array[Double], delta: Double): Array[Int] = ???

  /**
    * Fast single numerical variable neighbor entropy compute.
    *
    * @param data  1-dimension data.
    * @param delta The threshold of neighborhood relationshop.
    * @return neighbor entropy
    */
  def neighborEntropy(data: Array[Double], delta: Double): Double = ???

  /**
    * Estimate the delta-neighbor joint entropy. Use infinite norm as the measure of neighborhood relationship. The
    * ordering of elements not guaranteed.
    *
    * @param data    2-dimension data.
    * @param delta   The threshold of neighborhood relationship.
    * @param numBins Number of bins to estimate the count.
    * @return estimate of neighbor entropy.
    */
  def estimateNJE(data: Array[(Double, Double)], delta: Double, numBins: Int = 100000): Double = ???

  /**
    * Estimate the delta-neighbor counts of pairs. Use infinite norm as the measure of neighborhood relationship. The
    * ordering of elements not guaranteed.
    *
    * @param data    2-dimension data.
    * @param delta   The threshold of neighborhood relationship.
    * @param numBins Number of bins to estimate the count.
    * @return Approximate numbers of elements' neighbor.
    */
  def estimateNeighborCounts(data: Array[(Double, Double)], delta: Double, numBins: Int): Array[Int] = ???

}
