import java.io.*
import java.util.*

val isL = File("lib.kt").exists()

val d4x = intArrayOf(0,1,0,-1)
val d4y = intArrayOf(1,0,-1,0)
val d8x = intArrayOf(0,1,1,1,0,-1,-1,-1)
val d8y = intArrayOf(1,1,0,-1,-1,-1,0,1)

val imax = Int.MAX_VALUE
val imin = Int.MIN_VALUE
val lmax = Long.MAX_VALUE
val lmin = Long.MIN_VALUE

val umod = 1_000_000_007L
val cmod = 998_244_353L
val start = System.currentTimeMillis()

var n = 0
lateinit var p: IntArray

fun code() {
    
}

fun main() {

    var T = 1
    T = ini()

    repeat(T) { code() }
    isL.takeIf{it}?.let{ fh() }

    bw.flush()
    bw.close()
    br.close()
    System.gc()
}

// Additional I/O functions for competitive programming

fun ini(): Int = br.readLine().trim().toInt()
fun inl(): Long = br.readLine().trim().toLong()
fun ins(): String = br.readLine().trim()
fun inia(): IntArray = br.readLine().trim().split(" ").map { it.toInt() }.toIntArray()
fun inla(): LongArray = br.readLine().trim().split(" ").map { it.toLong() }.toLongArray()
fun insa(): Array<String> = br.readLine().trim().split(" ").toTypedArray()

fun pl(x: Any) { bw.write("${x.toString()} ") }
fun nl() { bw.write("\n") }

val br = when {
    isL -> BufferedReader(FileReader("inp.txt"))
    else-> BufferedReader(InputStreamReader(System.`in`))
}

val bw = when {
    isL -> BufferedWriter(FileWriter("opt.txt"))
    else-> BufferedWriter(OutputStreamWriter(System.out))
}

fun fh() {
    bw.write("\nTime: ${System.currentTimeMillis()-start} ms\n")
}