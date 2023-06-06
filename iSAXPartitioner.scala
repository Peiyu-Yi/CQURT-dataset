/**
  * === Fairness of Usage 
  * Please reference this paper in your paper as 
  * Liang Zhang, Noura Alghamdi, Mohamed Y. Eltabakh, Elke A. Rundensteiner. 
  * TARDIS: Distributed Indexing Framework for Big Time Series Data. 
  * Proceedings of 35th IEEE International Conference on Data Engineering ICDE, 2019.

  * ===  Disclaimer 
  * This program is copyright protected © 2018 by Liang Zhang, Noura Alghamdi, Mohamed Y. Eltabakh, Elke A. Rundensteiner. 
  * Data Science Research Group @ WPI - Worcester Polytechnic Institute.
  * Unless stated otherwise, all software is provided free of charge. As well, all software is provided on an "as is" basis 
  * without warranty of any kind, express or implied. Under no circumstances and under no legal theory, whether in tort, 
  * contract, or otherwise, shall Liang Zhang, Noura Alghamdi, Mohamed Y. Eltabakh or Elke A. Rundensteiner be liable to 
  * you or to any other person for any indirect, special, incidental, or consequential damages of any character including, 
  * without limitation, damages for loss of goodwill, work stoppage, computer failure or malfunction, or for any and all other damages or losses.
  * If you do not agree with these terms, then you you are advised to not use the software.
  */


package org.apache.spark.edu.wpi.dsrg.tardis.isax

/**
  * Created by leon on 8/12/17.
  */

import org.apache.spark.Partitioner

class iSAXPartitioner(tree: iSAXTreeHex) extends Partitioner {
  override def numPartitions: Int = tree.getPartitionNumber()

  override def getPartition(key: Any): Int = {
    tree.getPId(key.toString)
  }

  override def equals(other: scala.Any): Boolean = other match {
    case temp: iSAXPartitioner => temp.numPartitions == numPartitions
    case _ => false
  }
}

class iSAXPartitionerSampling(tree: iSAXTreeHex) extends Partitioner {
  private val extraId = tree.getPartitionNumber()
  override def numPartitions: Int = extraId +1

  override def getPartition(key: Any): Int = {
    try{
      tree.getPId(key.toString)
    }catch {
      case e:Exception => extraId
    }
  }

  override def equals(other: scala.Any): Boolean = other match {
    case temp: iSAXPartitioner => temp.numPartitions == numPartitions
    case _ => false
  }
}
