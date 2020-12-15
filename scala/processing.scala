import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import org.apache.spark.ml.linalg.{Vectors, Vector}

def getImageVec(w: Integer, h: Integer, img: String): Vector = {
  val photo = ImageIO.read(new File(img))
  Vectors.dense((0 until w * h).map(x => ((photo.getRGB(x % w, x / w) >> 16 & 0xFF) * 1.0)).to[Array])
}
