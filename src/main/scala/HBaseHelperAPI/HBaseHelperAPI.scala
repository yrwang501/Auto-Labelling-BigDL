package HBaseHelperAPI

import java.io.ByteArrayInputStream
import java.nio.charset.StandardCharsets

import org.apache.hadoop.fs.Path
import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client._
import org.apache.hadoop.hbase.util.Bytes

object HBaseHelperAPI {
  private var cachedTable : Table = _;
  def connectToHBase(coreSitePath: String, hbaseSitePath: String, hbaseTalbeName: String): Table = {
    val hbaseConf = HBaseConfiguration.create()
    hbaseConf.addResource(new Path(coreSitePath))
    hbaseConf.addResource(new Path(hbaseSitePath))

    val connection = ConnectionFactory.createConnection(hbaseConf)
    val table = connection.getTable(TableName.valueOf(hbaseTalbeName))
    table
  }

  def connectToHBaseWithConf(coreSiteConf: String, hbaseSiteConf: String, hbaseTalbeName: String): Table = {
    val hbaseConf = HBaseConfiguration.create()
    hbaseConf.addResource(new ByteArrayInputStream(coreSiteConf.getBytes(StandardCharsets.UTF_8)))
    hbaseConf.addResource(new ByteArrayInputStream(hbaseSiteConf.getBytes(StandardCharsets.UTF_8)))

    val connection = ConnectionFactory.createConnection(hbaseConf)
    val table = connection.getTable(TableName.valueOf(hbaseTalbeName))
    table
  }

  def connectToHBaseCached(coreSiteConfcoreSiteConf: String, hbaseSiteConf: String, hbaseTalbeName: String): Table = {
    if(cachedTable==null){
      cachedTable=connectToHBaseWithConf(hbaseSiteConf,hbaseSiteConf,hbaseTalbeName)
    }
    cachedTable
  }

  def clearCachedConnection() = {
    cachedTable.close()
    cachedTable=null
  }

  def getData(table: Table, rowKey: Array[Byte], colFamily: Array[Byte], colQualifier: Array[Byte]): String = {
    val getObj = new Get(rowKey)
    getObj.addColumn(colFamily, colQualifier)

    val result = table.get(getObj)
    val value = result.getValue(colFamily, colQualifier)
    Bytes.toString(value)
  }

  def putData(table: Table, rowKey: Array[Byte], colFamily: Array[Byte], colQualifier: Array[Byte], value: Array[Byte]): Unit = {
    val putObj = new Put(rowKey)
    putObj.addColumn(colFamily, colQualifier, value)

    try {
      table.put(putObj)
    } catch {
      case _ => println("Failed to put data into HBase!")
    }

    println("Successfully uploaded data into HBase table.")
  }


  def scanGetData(table: Table, startRow: Array[Byte], stopRow: Array[Byte], length: Int, colFamily: Array[Byte], colQualifiers: Array[Array[Byte]]): Array[String] = {
    val numQualifiers = colQualifiers.length
    val scanObj = new Scan(startRow, stopRow)
    scanObj.addColumn(colFamily, Bytes.toBytes("train"))
    for (i <- 0 until numQualifiers){
      scanObj.addColumn(colFamily, colQualifiers(i))
    }

    val scanner = table.getScanner(scanObj)
    var resultArray = new Array[Result](length)
    try {
      resultArray = scanner.next(length)
    } catch {
      case _ => println("Failed to read from the scanner of the HBase table")
    } finally {
      scanner.close()
    }

    // return an array of, separated by semicolons, strings which are the columns you want to retrieve
    resultArray.filter(result => {
      val train = result.getValue(colFamily, Bytes.toBytes("train"))
      val trainTest = Bytes.toString(train)
      trainTest == "0"
    }).map(result => {
      var dataStr = Bytes.toString(result.getRow()) + ";"
      for (i <- 0 until numQualifiers) {
        val bytes = result.getValue(colFamily, colQualifiers(i))
        dataStr = dataStr + Bytes.toString(bytes) + ";"
      }
      dataStr
    })
  }

  def scanGetFullData(table: Table, startRow: Array[Byte], stopRow: Array[Byte], length: Int, colFamily: Array[Byte], colQualifiers: Array[Array[Byte]]): Array[String] = {
    val numQualifiers = colQualifiers.length
    val scanObj = new Scan(startRow, stopRow)
    scanObj.addColumn(colFamily, Bytes.toBytes("train"))
    for (i <- 0 until numQualifiers){
      scanObj.addColumn(colFamily, colQualifiers(i))
    }

    val scanner = table.getScanner(scanObj)
    var resultArray = new Array[Result](length)
    try {
      resultArray = scanner.next(length)
    } catch {
      case _ => println("Failed to read from the scanner of the HBase table")
    } finally {
      scanner.close()
    }

    resultArray.map(result => {
      var dataStr = Bytes.toString(result.getRow()) + ";"
      for (i <- 0 until numQualifiers) {
        val bytes = result.getValue(colFamily, colQualifiers(i))
        dataStr = dataStr + Bytes.toString(bytes) + ";"
      }
      dataStr
    })
  }


  // examples in main function
  def main(args: Array[String]): Unit = {
    val table = connectToHBase("/home/yilinma/Documents/IntelliJ_IDEA_Projects/AIMasterTrainingCrop/core-site.xml", "/home/yilinma/Documents/IntelliJ_IDEA_Projects/AIMasterTrainingCrop/hbase-site.xml", "kfb_512_100_test")

//    val row1 = Bytes.toBytes("%09d".format(11))
//
//    val family = Bytes.toBytes("123_s20")
//    val qualifier = Bytes.toBytes("offset")
//    val value = Bytes.toBytes("#")
//
//    putData(table, row1, family, qualifier, value)
//    val label1 = getData(table, row1, family, qualifier)
//    println(label1)

    val startRow = Bytes.toBytes("000000001")
    val stopRow = Bytes.toBytes("%09d".format(481))

    val scanFamily = Bytes.toBytes("123_s20")
    val qualifiers = Array(Bytes.toBytes("train"), Bytes.toBytes("pos"), Bytes.toBytes("offset"))

    val labelArray = scanGetData(table, startRow, stopRow, 480, scanFamily, qualifiers)

    labelArray.map(label => println(label))

    table.close()
  }
}
