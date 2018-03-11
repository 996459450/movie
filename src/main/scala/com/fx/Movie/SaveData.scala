package com.fx.Movie

import java.util.Properties
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SaveMode


object SaveData {
  def main(args: Array[String]): Unit = {
//    val mysql_username = "root";
//    val mysql_pwd = "briup"
//    val mysql_con = "jdbc:mysql://192.168.122.168:3306/movie"
    val sc = new SparkContext(new SparkConf().setAppName("save data to db")
        .setMaster("spark://192.168.122.168:7077"))
    val sqlContext = new SQLContext(sc) 
    val userDF = sqlContext.read.json("hdfs://192.168.122.168:9000/data/user.json");
//    val properties = new Properties();
//    properties.put("user", mysql_username)
//    properties.put("passwd", mysql_pwd)
    
//    userDF.show();
//    userDF.write.mode(SaveMode.Append).jdbc(mysql_con, "data", properties)
    userDF.write.mode(SaveMode.Append).json("hdfs://192.168.122.168:9000/user/movie")
  }
}