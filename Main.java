/* package codechef; // don't place package name! */

import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
class Main
{
    static BufferedWriter put = new BufferedWriter(new OutputStreamWriter(System.out));
    static int intmax = Integer.MAX_VALUE;
    static int intmin = Integer.MIN_VALUE;
    static long mod = 1000000007L;
    
    public static void main (String[] args) throws java.lang.Exception
	{
	    FastReader get = new FastReader();
	    
	    int T = get.nextInt();
	    
	    while(T-->0)
	    {
	    	int n=get.nextInt();
	    	
	    	long [] arr= readArr2(n, get);
	        
	        ans( arr, n );
	        
	        put.flush();
	    }
	    
	    put.close();
	}
	
	public static void ans( long [] arr, int n ) throws java.lang.Exception
	{
		
	}
	
	//
	// FUNCTIONS BELOW :
	// isPrime(), lcm(), gcd(), totient(), findDiv(), sortInt(), sortLong(),
	// power(), freqArr(), MATRIX: multiply(), power()
	// CLASSES BELOW :
	// DSU, FenwickTree, SegmentTree, LazySegTree, RangeBit, SparseTable, LCA, BitSet, MaxFlow
	//
	
	public static int[] readArr(int N, FastReader get) throws Exception
    {
        int[] arr = new int[N];
        StringTokenizer st = new StringTokenizer(get.nextLine(), " ");
        for(int i=0; i < N; i++) arr[i] = Integer.parseInt(st.nextToken());
        
        return arr;
    }
    
    public static long[] readArr2(int N, FastReader get) throws Exception
    {
        long[] arr = new long[N];
        StringTokenizer st = new StringTokenizer(get.nextLine(), " ");
        for(int i=0; i < N; i++) arr[i] = Long.parseLong(st.nextToken());
        
        return arr;
    }
    
    public static int[][] readMat(int N, int M, FastReader get) throws Exception
    {
        int [][] mat = new int[N][M];
        
        for(int i=0; i < N; i++)
        {
            StringTokenizer st = new StringTokenizer(get.nextLine(), " ");
            for(int j=0; j < M; j++) mat[i][j] = Integer.parseInt(st.nextToken());
        }
        
        return mat;
    }
    
    public static void print(int[] arr) throws java.lang.Exception
    {
        for(int x: arr)
            put.write(x+" ");
        put.write("\n");
    }
    
    public static void printMat(int[][] arr) throws java.lang.Exception
    {
        for(int i=0; i < arr.length; i++)
        {
            for(int j=0; j < arr[0].length; j++)
            {
                put.write(arr[i][j] + " ");
            }
            put.write("\n");
        }
        put.write("\n");
    }
	
    public static boolean isPrime(long n)
    {
        if(n < 2) return false;
        if(n == 2 || n == 3) return true;
        if(n%2 == 0 || n%3 == 0) return false;
        long sqrtN = (long)Math.sqrt(n)+1;
        for(long i = 6L; i <= sqrtN; i += 6) {
            if(n%(i-1) == 0 || n%(i+1) == 0) return false;
        }
        return true;
    }
    
    public static long lcm(long a, long b)
    {
        return (a / gcd(a, b)) * b;
    }
    
    public static long gcd(long a, long b)
    {
        if(a > b) a = (a+b)-(b=a);
        if(a == 0L) return b;
        
        return gcd(b%a, a);
    }
    
    public static long totient(long n)
    {
        long result = n;
        for (int p = 2; p*p <= n; ++p)
            if (n % p == 0)
            {
                while(n%p == 0)
                    n /= p;
                result -= result/p;
            }
        if (n > 1) result -= result/n;
        
        return result;
    }
    
    public static ArrayList<Integer> findDiv(int N)
    {
        //gets all divisors of N
        ArrayList<Integer> ls1 = new ArrayList<Integer>();
        ArrayList<Integer> ls2 = new ArrayList<Integer>();
        for(int i=1; i <= (int)(Math.sqrt(N)+0.00000001); i++)
            if(N%i == 0)
            {
                ls1.add(i);
                ls2.add(N/i);
            }
        Collections.reverse(ls2);
        for(int b: ls2)
            if(b != ls1.get(ls1.size()-1))
                ls1.add(b);
        return ls1;
    }
    
    public static void sortInt(int[] arr)
    {
        //because Arrays.sort() uses quicksort which is dumb
        //Collections.sort() uses merge sort
        
        ArrayList<Integer> ls = new ArrayList<Integer>();
        
        for(int x: arr) ls.add(x);
        
        Collections.sort(ls);
        for(int i=0; i < arr.length; i++) arr[i] = ls.get(i);
    }
    
    public static void sortLong(long[] arr)
    {
        //because Arrays.sort() uses quicksort which is dumb
        //Collections.sort() uses merge sort
        
        ArrayList<Long> ls = new ArrayList<Long>();
        
        for(long x: arr) ls.add(x);
        
        Collections.sort(ls);
        for(int i=0; i < arr.length; i++) arr[i] = ls.get(i);
    }
    
    public static void sortChar(char[] arr)
    {
        //because Arrays.sort() uses quicksort which is dumb
        //Collections.sort() uses merge sort
        
        ArrayList<Character> ls = new ArrayList<Character>();
        
        for(char x: arr) ls.add(x);
        
        Collections.sort(ls);
        for(int i=0; i < arr.length; i++) arr[i] = ls.get(i);
    }
    
    public static String sortString(String inputString)
    {
        char tempArray[] = inputString.toCharArray();
        sortChar(tempArray);
        return new String(tempArray);
    }
    
    public static long power(long x, long y, long p)
    {
        //0^0 = 1
        long res = 1L;
        x = x%p;
        while(y > 0)
        {
            if((y&1)==1)
                res = (res*x)%p;
            y >>= 1;
            x = (x*x)%p;
        }
        return res;
    }
    
    //custom multiset (replace with HashMap if needed)
    public static void push(TreeMap<Integer, Integer> map, int k, int v)
    {
        //map[k] += v;
        if(!map.containsKey(k)) map.put(k, v);
        else map.put(k, map.get(k)+v);
    }
    
    public static void pull(TreeMap<Integer, Integer> map, int k, int v)
    {
        //assumes map[k] >= v
        //map[k] -= v
        int lol = map.get(k);
        
        if(lol == v) map.remove(k);
        else map.put(k, lol-v);
    }
    
    public static HashMap<Integer, Integer> freqMap(int[] arr)
    {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        
        for(int x: arr)
            if(!map.containsKey(x)) map.put(x, map.get(x) + 1);
            else map.put(x, 1);
            
        return map;
    }
    
    public static long[][] multiply(long[][] left, long[][] right)
    {
        long MOD = 1000000007L;
        int N = left.length;
        int M = right[0].length;
        long[][] res = new long[N][M];
        for(int a=0; a < N; a++)
            for(int b=0; b < M; b++)
                for(int c=0; c < left[0].length; c++)
                {
                    res[a][b] += (left[a][c]*right[c][b])%MOD;
                    if(res[a][b] >= MOD)
                        res[a][b] -= MOD;
                }
        return res;
    }
    
    public static long[][] power(long[][] grid, long pow)
    {
        long[][] res = new long[grid.length][grid[0].length];
        for(int i=0; i < res.length; i++)
            res[i][i] = 1L;
        long[][] curr = grid.clone();
        while(pow > 0)
        {
            if((pow&1L) == 1L)
                res = multiply(curr, res);
            pow >>= 1;
            curr = multiply(curr, curr);
        }
        return res;
    }
}

class Pair implements Comparable<Pair>
{
    int a, b;
    
    public Pair(int a, int b)
    {
        this.a = a;
        this.b = b;
    }
    
    @Override
    public int compareTo(Pair p)
    {
    	if(this.b > p.b) return 1;
    	else return -1;
    }
}

class DSU
{
    public int[] dsu;
    public int[] size;
 
    public DSU(int N)
    {
        dsu = new int[N+1];
        size = new int[N+1];
        for(int i=0; i <= N; i++)
        {
            dsu[i] = i;
            size[i] = 1;
        }
    }
    //with path compression, no find by rank
    public int find(int x)
    {
        return dsu[x] == x ? x : (dsu[x] = find(dsu[x]));
    }
    public void merge(int x, int y)
    {
        int fx = find(x);
        int fy = find(y);
        dsu[fx] = fy;
    }
    public void merge(int x, int y, boolean sized)
    {
        int fx = find(x);
        int fy = find(y);
        size[fy] += size[fx];
        dsu[fx] = fy;
    }
}

class FenwickTree
{
    //Binary Indexed Tree
    //1 indexed
    public int[] tree;
    public int size;
 
    public FenwickTree(int size)
    {
        this.size = size;
        tree = new int[size+5];
    }
    public void add(int i, int v)
    {
        while(i <= size)
        {
            tree[i] += v;
            i += i&-i;
        }
    }
    public int find(int i)
    {
        int res = 0;
        while(i >= 1)
        {
            res += tree[i];
            i -= i&-i;
        }
        return res;
    }
    public int find(int l, int r)
    {
        return find(r)-find(l-1);
    }
}

class SegmentTree
{
    //Tlatoani's segment tree
    //iterative implementation = low constant runtime factor
    //range query, non lazy
    final int[] val;
    final int treeFrom;
    final int length;
 
    public SegmentTree(int treeFrom, int treeTo)
    {
        this.treeFrom = treeFrom;
        int length = treeTo - treeFrom + 1;
        int l;
        for (l = 0; (1 << l) < length; l++);
        val = new int[1 << (l + 1)];
        this.length = 1 << l;
    }
    public void update(int index, int delta)
    {
        //replaces value
        int node = index - treeFrom + length;
        val[node] = delta;
        for (node >>= 1; node > 0; node >>= 1)
            val[node] = comb(val[node << 1], val[(node << 1) + 1]);
    }
    public int query(int from, int to)
    {
        //inclusive bounds
        if (to < from)
            return 0; //0 or 1?
        from += length - treeFrom;
        to += length - treeFrom + 1;
        //0 or 1?
        int res = 0;
        for (; from + (from & -from) <= to; from += from & -from)
            res = comb(res, val[from / (from & -from)]);
        for (; to - (to & -to) >= from; to -= to & -to)
            res = comb(res, val[(to - (to & -to)) / (to & -to)]);
        return res;
    }
    public int comb(int a, int b)
    {
        //change this
        return Math.max(a,b);
    }
}

class LazySegTree
{
    //definitions
    private int NULL = -1;
    private int[] tree;
    private int[] lazy;
    private int length;
 
    public LazySegTree(int N)
    {
        length = N;   int b;
        for(b=0; (1<<b) < length; b++);
        tree = new int[1<<(b+1)];
        lazy = new int[1<<(b+1)];
    }
    public int query(int left, int right)
    {
        //left and right are 0-indexed
        return get(1, 0, length-1, left, right);
    }
    private int get(int v, int currL, int currR, int L, int R)
    {
        if(L > R)
            return NULL;
        if(L <= currL && currR <= R)
            return tree[v];
        propagate(v);
        int mid = (currL+currR)/2;
        return comb(get(v*2, currL, mid, L, Math.min(R, mid)),
                get(v*2+1, mid+1, currR, Math.max(L, mid+1), R));
    }
    public void update(int left, int right, int delta)
    {
        add(1, 0, length-1, left, right, delta);
    }
    private void add(int v, int currL, int currR, int L, int R, int delta)
    {
        if(L > R)
            return;
        if(currL == L && currR == R)
        {
            //exact covering
            tree[v] += delta;
            lazy[v] += delta;
            return;
        }
        propagate(v);
        int mid = (currL+currR)/2;
        add(v*2, currL, mid, L, Math.min(R, mid), delta);
        add(v*2+1, mid+1, currR, Math.max(L, mid+1), R, delta);
        tree[v] = comb(tree[v*2], tree[v*2+1]);
    }
    private void propagate(int v)
    {
        //tree[v] already has lazy[v]
        if(lazy[v] == 0)
            return;
        tree[v*2] += lazy[v];
        lazy[v*2] += lazy[v];
        tree[v*2+1] += lazy[v];
        lazy[v*2+1] += lazy[v];
        lazy[v] = 0;
    }
    private int comb(int a, int b)
    {
        return Math.max(a,b);
    }
}

class RangeBit
{
    //FenwickTree and RangeBit are faster than LazySegTree by constant factor
    final int[] value;
    final int[] weightedVal;
 
    public RangeBit(int treeTo)
    {
        value = new int[treeTo+2];
        weightedVal = new int[treeTo+2];
    }
    private void updateHelper(int index, int delta)
    {
        int weightedDelta = index*delta;
        for(int j = index; j < value.length; j += j & -j)
        {
            value[j] += delta;
            weightedVal[j] += weightedDelta;
        }
    }
    public void update(int from, int to, int delta)
    {
        updateHelper(from, delta);
        updateHelper(to + 1, -delta);
    }
    private int query(int to)
    {
        int res = 0;
        int weightedRes = 0;
        for (int j = to; j > 0; j -= j & -j)
        {
            res += value[j];
            weightedRes += weightedVal[j];
        }
        return ((to + 1)*res)-weightedRes;
    }
    public int query(int from, int to)
    {
        if (to < from)
            return 0;
        return query(to) - query(from - 1);
    }
}

class SparseTable
{
    public int[] log;
    public int[][] table;
    public int N;  public int K;
 
    public SparseTable(int N)
    {
        this.N = N;
        log = new int[N+2];
        K = Integer.numberOfTrailingZeros(Integer.highestOneBit(N));
        table = new int[N][K+1];
        sparsywarsy();
    }
    private void sparsywarsy()
    {
        log[1] = 0;
        for(int i=2; i <= N+1; i++)
            log[i] = log[i/2]+1;
    }
    public void lift(int[] arr)
    {
        int n = arr.length;
        for(int i=0; i < n; i++)
            table[i][0] = arr[i];
        for(int j=1; j <= K; j++)
            for(int i=0; i + (1 << j) <= n; i++)
                table[i][j] = Math.min(table[i][j-1], table[i+(1 << (j - 1))][j-1]);
    }
    public int query(int L, int R)
    {
        //inclusive, 1 indexed
        L--;  R--;
        int mexico = log[R-L+1];
        return Math.min(table[L][mexico], table[R-(1 << mexico)+1][mexico]);
    }
}

class LCA
{
    public int N, root;
    public ArrayDeque<Integer>[] edges;
    private int[] enter;
    private int[] exit;
    private int LOG = 17; //change this
    private int[][] dp;
 
    public LCA(int n, ArrayDeque<Integer>[] edges, int r)
    {
        N = n;   root = r;
        enter = new int[N+1];
        exit = new int[N+1];
        dp = new int[N+1][LOG];
        this.edges = edges;
        int[] time = new int[1];
        //change to iterative dfs if N is large
        dfs(root, 0, time);
        dp[root][0] = 1;
        for(int b=1; b < LOG; b++)
            for(int v=1; v <= N; v++)
                dp[v][b] = dp[dp[v][b-1]][b-1];
    }
    private void dfs(int curr, int par, int[] time)
    {
        dp[curr][0] = par;
        enter[curr] = ++time[0];
        for(int next: edges[curr])
            if(next != par)
                dfs(next, curr, time);
        exit[curr] = ++time[0];
    }
    public int lca(int x, int y)
    {
        if(isAnc(x, y))
            return x;
        if(isAnc(y, x))
            return y;
        int curr = x;
        for(int b=LOG-1; b >= 0; b--)
        {
            int temp = dp[curr][b];
            if(!isAnc(temp, y))
                curr = temp;
        }
        return dp[curr][0];
    }
    private boolean isAnc(int anc, int curr)
    {
        return enter[anc] <= enter[curr] && exit[anc] >= exit[curr];
    }
}

class BitSet
{
    private int CONS = 62; //safe
    public long[] sets;
    public int size;
 
    public BitSet(int N)
    {
        size = N;
        if(N%CONS == 0)
            sets = new long[N/CONS];
        else
            sets = new long[N/CONS+1];
    }
    public void add(int i)
    {
        int dex = i/CONS;
        int thing = i%CONS;
        sets[dex] |= (1L << thing);
    }
    public int and(BitSet oth)
    {
        int boof = Math.min(sets.length, oth.sets.length);
        int res = 0;
        for(int i=0; i < boof; i++)
            res += Long.bitCount(sets[i] & oth.sets[i]);
        return res;
    }
    public int xor(BitSet oth)
    {
        int boof = Math.min(sets.length, oth.sets.length);
        int res = 0;
        for(int i=0; i < boof; i++)
            res += Long.bitCount(sets[i] ^ oth.sets[i]);
        return res;
    }
}

class MaxFlow
{
    //Dinic with optimizations (see magic array in dfs function)
    public int N, source, sink;
    public ArrayList<Edge>[] edges;
    private int[] depth;
 
    public MaxFlow(int n, int x, int y)
    {
        N = n;
        source = x;
        sink = y;
        edges = new ArrayList[N+1];
        for(int i=0; i <= N; i++)
            edges[i] = new ArrayList<Edge>();
        depth = new int[N+1];
    }
    public void addEdge(int from, int to, long cap)
    {
        Edge forward = new Edge(from, to, cap);
        Edge backward = new Edge(to, from, 0L);
        forward.residual = backward;
        backward.residual = forward;
        edges[from].add(forward);
        edges[to].add(backward);
    }
    public long mfmc()
    {
        long res = 0L;
        int[] magic = new int[N+1];
        while(assignDepths())
        {
            long flow = dfs(source, Long.MAX_VALUE/2, magic);
            while(flow > 0)
            {
                res += flow;
                flow = dfs(source, Long.MAX_VALUE/2, magic);
            }
            magic = new int[N+1];
        }
        return res;
    }
    private boolean assignDepths()
    {
        Arrays.fill(depth, -69);
        ArrayDeque<Integer> q = new ArrayDeque<Integer>();
        q.add(source);
        depth[source] = 0;
        while(q.size() > 0)
        {
            int curr = q.poll();
            for(Edge e: edges[curr])
                if(e.capacityLeft() > 0 && depth[e.to] == -69)
                {
                    depth[e.to] = depth[curr]+1;
                    q.add(e.to);
                }
        }
        return depth[sink] != -69;
    }
    private long dfs(int curr, long bottleneck, int[] magic)
    {
        if(curr == sink)
            return bottleneck;
        for(; magic[curr] < edges[curr].size(); magic[curr]++)
        {
            Edge e = edges[curr].get(magic[curr]);
            if(e.capacityLeft() > 0 && depth[e.to]-depth[curr] == 1)
            {
                long val = dfs(e.to, Math.min(bottleneck, e.capacityLeft()), magic);
                if(val > 0)
                {
                    e.augment(val);
                    return val;
                }
            }
        }
        return 0L;  //no flow
    }
    private class Edge
    {
        public int from, to;
        public long flow, capacity;
        public Edge residual;
 
        public Edge(int f, int t, long cap)
        {
            from = f;
            to = t;
            capacity = cap;
        }
        public long capacityLeft()
        {
            return capacity-flow;
        }
        public void augment(long val)
        {
            flow += val;
            residual.flow -= val;
        }
    }
}

class FastReader
{
    BufferedReader br;
    StringTokenizer st;

    public FastReader()
    {
        br = new BufferedReader(new InputStreamReader(System.in));
    }

    String next()
    {
        while (st == null || !st.hasMoreElements())
        {
            try
            {
                st = new StringTokenizer(br.readLine());
            }
            catch (IOException  e)
            {
                e.printStackTrace();
            }
        }
        return st.nextToken();
    }

    int nextInt()
    {
        return Integer.parseInt(next());
    }

    long nextLong()
    {
        return Long.parseLong(next());
    }

    double nextDouble()
    {
        return Double.parseDouble(next());
    }

    String nextLine()
    {
        String str = "";
        try
        {
            str = br.readLine();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        return str;
    }
}