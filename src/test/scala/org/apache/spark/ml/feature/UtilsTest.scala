package org.apache.spark.ml.feature

import org.apache.spark.ml.feature.Utils._
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class UtilsTest extends FunSuite with BeforeAndAfterAll{
  var values: Array[(Double, Double)] = _

  override protected def beforeAll(): Unit = {
    val num = 10000
    values = (1 to num).toArray.map(_ => (util.Random.nextDouble(), util.Random.nextDouble())).sortBy(_._1)
  }

  test("test the precision of estimate mutual information should small enough"){
    val (x, y) = values.unzip
    val delta = 0.1
    val ex = neighborEntropy(x, delta)
    val ey = neighborEntropy(y, delta)
    val estimateJointEntropy = estimateNJE(values, delta)
    val estimate = ex + ey - estimateJointEntropy
    val exact = exactNMI(values,delta)
    assert(math.abs(estimate - exact) < 0.00001)
  }
}
