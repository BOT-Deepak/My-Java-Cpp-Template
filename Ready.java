import java.util.*;
import java.io.*;
import java.lang.*;
import java.util.stream.Collectors;

public class Library {

    static int d4[][] = new int[][] {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    static int d8[][] = new int[][] {{0, -1}, {0, 1}, {-1, 0}, {1, 0}, {1, -1}, {1, 1}, {-1, -1}, {-1, 1}};
    static int imax = Integer.MAX_VALUE, imin = Integer.MIN_VALUE;
    static long lmax = Long.MAX_VALUE, lmin = Long.MIN_VALUE;
    static long umod = (long)1e9+7, cmod = 998244353;


    // checks if n is prime number or not
    private static boolean is_prime(long n) {
        if(n < 2) return false;
        if(n == 2) return true;
        if(n%2 == 0) return false;

        long sq = (long)Math.sqrt(n);
        for(int i=3; i<=sq; i+=2)
            if(n%i == 0)
                return false;

        return true;
    }

    // greatest common divisor (hcf)
    private static long get_gcd(long x, long y) {
        if(x == 0) return y;
        return get_gcd(y%x, x);
    }

    // lowest common multiple (lcm)
    private static long get_lcm(long x, long y) {
        return (x / get_gcd(x, y)) * y;
    }

    // count of numbers from 1 to n-1 which are relatively prime with n
    private static long get_totient(long n) {
        long res = n;
        for(long p=2; p*p<=n; ++p) {
            if(n%p == 0) {
                while(n%p == 0) n /= p;
                res *= (1.0 - (1.0 / double(p)));
            }}

        if(n > 1) res -= (res/n);
        return res;
    }

    // x power y mod p
    private static long get_power(long x, long y, long p) {
        long res = 1L;
        x %= p;
        while(y > 0) {
            if((y&1) == 1) res = (res*x)%p;
            y >>= 1;
            x = (x*x)%p;
        }
        return res;
    }

    private static void get_digpow(long x, long y, int n) {
        long pw = y * (long)Math.log10(x);
        String fd = Math.pow(10, pw-Math.floor(pw)) * Math.pow(10, n-1) + "";
        String ld = get_power(x, y, Math.pow(10, n));
    }
}

class RabinKarp {
    public long[] pp;

    public RabinKarp(n) {
        this.pp = new long[n];
        pp[0] = 1;
        pp[1] = 31;
        for(int i=2; i<n; ++i) pp[i] = (pp[i-1]*31)%umod;
    }

    public long get_string_hash(String s) {
        long res = 0;
        for(int i=0; i<s.length(); ++i)
            res = (res + (s.charAt(i)-'a'+1)*pp[i])%umod;
        return res;
    }

    public long get_char_hash(char c, int i) {
        return ((c-'a'+1)*pp[i])%umod;
    }
}

class FactorSeive {

    public int N;
    public int[] SPF;
    public List<Integer> PRIMES;
    public Map<Integer, Integer> PX;

    public FactorSeive(int n) {
        this.N = n;
    }

    private List<Integer> get_factors() {
        
        List<Integer> res = new ArrayList<>();
        int i = 1;
        
        while(i*i < N) {
            if(N % i == 0) res.add(i);
            ++i;}

        if(i-(N/i) == 1) --i;

        while(i > 0) {
            if(N % i == 0) res.append(N/i);
            --i;}

        return res;
    }

    private void prime_sieve() {

        this.PRIMES = new ArrayList<>();
        int[] p = new int[N+1];

        for(int i=2; i<=N; ++i)
            if(p[i] == 0)
                for(int j=i*i; i<=N; j+=i)
                    p[j] = 1;

        for(int i=2; i<=N; ++i)
            if(p[i] == 0)
                PRIMES.add(i);
    }

    // sieve to get prime factors in O(logN)
    private static void prime_factor_seive() {
        this.SPF = new int[N+1];
        SPF[1] = 1;

        for(int i=2; i<=N; ++i) SPF[i] = i;
        for(int i=4; i<=N; i+=2) SPF[i] = 2;

        for(int i=3; i*i<=N; ++i)
            if(SPF[i] == i)
                for(int j=i*i; j<=N; j+=i)
                    if(SPF[j] == j) SPF[j] = i;
    }

    // count the total number of factors of n and get P1^x1, P2^x2 for every n
    private static int count_factors() {
        this.PX = new HashMap<>();
        int res = 1, n = N;

        while(n > 1) {
            int cur = SPF[n], cnt = 0;
            while(n > 1 && SPF[n] == cur) {
                ++cnt;
                n /= cur;
            }
            res *= (cnt+1);
            PX.put(cur, PX.getOrDefault(cur,0) + cnt);
        }

        return res;
    }
}

class StringMatching {

    public char[] S;
    public int[] PI;
    public int lenP, lenT, lenS;

    public StringMatching(String pattern, String haystack) {
        this.lenP = pattern.length();
        this.lenT = haystack.length();
        this.lenS = lenP + lenT + 1;
        this.PI = new int[lenS];
        this.S = (pattern + "#" + haystack).toCharArray();
    }

    private static List<Integer> get_kmp() {

        for(int i=1; i < lenS; ++i) {
            int j = PI[i-1];
            while(j > 0 && S[i] != S[j]) j = PI[j-1];
            if(S[i] == S[j]) ++j;
            PI[i] = j;
        }

        List<Integer> res = new ArrayList<>();
        for(int i=lenP+1; i < lenS; ++i)
            if(PI[i] == lenP) res.add(i-2*lenP);
        
        return res;
    }

    private static List<Integer> get_zalgo() {

        int L = 0, R = 0;
        for(int i=1; i < lenS; ++i) {

            if(i <= R && PI[i-L] < R-i+1)
                PI[i] = PI[i-L];
            else {
                if(i > R) L = R = i;
                else L = i;

                while(R < lenS && S[R-L] == S[R]) ++R;
                PI[i] = (R--)-L;
            }}

        List<Integer> res = new ArrayList<>();
        for(int i=0; i < lenS; ++i)
            if(PI[i] == lenP && i-lenP-1 <= lenS-lenP)
                res.add(i-lenP-1);
        
        return res;
    }
}

class FenvickTree {

    public int[] FEN;
    public int N;

    public FenvickTree(int n) {
        this.N = n+5;
        this.FEN = new int[N];
    }

    public void add(int ind, int val) {
        while(ind <= N) {
            FEN[ind] += val;
            ind += (ind & -ind);
        }
    }

    public int find(int ind) {
        int res = 0;
        while(ind > 0) {
            res += FEN[ind];
            ind -= (ind & -ind);
        }
        return res;
    }

    public int find(int left, int right) {
        return find(right) - find(left-1);
    }
}

class DSU {

    public int N;
    public int[] SIZE, PARENT;

    public DSU(int n) {
        this.N = n;
        this.SIZE = new int[n];
        this.PARENT = new int[n];

        for(int i=0; i < n; ++i) {
            SIZE[i] = 1;
            PARENT[i] = i;}
    }

    public int find(int x) {
        if(PARENT[x] != x)
            PARENT[x] = find(PARENT[x]);
        return PARENT[x];
    }

    public void merge(int x, int y) {
        int a = PARENT[x], b = PARENT[y];
        if(a == b) return;

        if(SIZE[a] < SIZE[b]) {
            a = a^b; b = a^b; a = a^b;}

        PARENT[b] = a;
        SIZE[a] += SIZE[b];
    }
}

class TrieNode {

    public int ALPHABET_SIZE = 26;
    public TrieNode[] CHILDREN;
    public boolean IS_END;

    public TrieNode() {
        this.CHILDREN = new TrieNode[ALPHABET_SIZE];
        this.IS_END = false;
    }
}

class Trie {

    public TrieNode ROOT;

    public Trie() {
        this.ROOT = new TrieNode();
    }

    public void insert(String s) {
        TrieNode pCrawl = this.ROOT;

        for(int i=0; i < s.length(); ++i) {
            int idx = s.charAt(i)-'a';
            
            if(pCrawl.CHILDREN[idx] == null) pCrawl.CHILDREN[idx] = new TrieNode();
            pCrawl = pCrawl.CHILDREN[idx];
        }

        pCrawl.IS_END = true;
    }

    public boolean search(String s) {
        TrieNode pCrawl = this.ROOT;

        for(int i=0; i < s.length(); ++i) {
            int idx = s.charAt(i)-'a';
            
            if(pCrawl.CHILDREN[idx] == null) return false;
            pCrawl = pCrawl.CHILDREN[idx];
        }

        return pCrawl.IS_END;
    }
}

class SegmentTree {

    public int[] TREE;
    public int N;

    public SegmentTree(int[] a) {
        this.N = a.length+5;
        this.TREE = new int[2*N];

        for(int i=0; i<N; ++i) TREE[N+i] = a[i];
        for(int i=N-1; i > 0; --i) TREE[i] = TREE[i<<1] + TREE[i<<1 | 1];
    }

    public void update(int p, int val) {
        for(TREE[p+=N] = val; p > 1; p>>=1) TREE[p>>1] = TREE[p] + TREE[p^1];
    }

    public int query(int l, int r) {
        int res = 0;
        for(l+=N, r+=N; l < r; l>>=1, r>>=1) {
            if((l&1) > 0) res += TREE[l++];
            if((r&1) > 0) res += TREE[--r];
        }
        return res;
    }
}

class LazySegmentTree {

    public int[] TREE;
    public int[] LAZY;
    public int N, H;

    public SegmentTree(int[] a) {
        this.N = a.length+5;
        this.H = Integer.SIZE - Integer.numberOfLeadingZeroes(N);
        this.TREE = new int[2*N];
        this.LAZY = new int[2*N];

        for(int i=0; i<N; ++i) TREE[N+i] = a[i];
        for(int i=N-1; i > 0; --i) TREE[i] = TREE[i<<1] + TREE[i<<1 | 1];
    }

    public void apply(int p, int val) {
        TREE[p] += val;
        if(p < N) LAZY[p] += val;
    }

    public void build(int p) {
        while(p > 1) {
            p>>=1;
            TREE[p] = Math.max(TREE[p<<1], TREE[p<<1 | 1]) + LAZY[p];
        }
    }

    public void push(int p) {
        for(int s = H; s > 0; --s) {
            int i = p >> s;

            if(LAZY[i] != 0) {
                apply(i<<1, LAZY[i]);
                apply(i<<1 | 1, LAZY[i]);
                LAZY[i] = 0;
            }
        }
    }

    public void update(int l, int r, int val) {
        l+=N; r+=N;
        int l0 = l, r0 = r;
        for(; l < r; l>>=1, r>>=1) {
            if((l&1) > 0) apply(l++, val);
            if((r&1) > 0) apply(--r, val);
        }
        build(l0); build(r0-1);
    }

    public int query(int l, int r) {
        l+=N; r+=N;
        push(l); push(r-1);

        int res = -(int)2e9;
        for(; l < r; l>>=1, r>>=1) {
            if((l&1) > 0) res = Math.max(res, TREE[l++]);
            if((r&1) > 0) res = Math.max(TREE[--r], res);
        }
        return res;
    }
}

class Aho {

    public Aho[] CHILDREN;
    public int PATTERN_INDEX;
    public Aho SUFFIX_LINK, OUTPUT_LINK;

    public Aho() {
        CHILDREN = new Aho[26];
        SUFFIX_LINK = null;
        OUTPUT_LINK = null;
        PATTERN_INDEX = Integer.MAX_VALUE;
    }
}

class Corasick {
    public Aho ROOT;
    
    public Corasick() {
        this.ROOT = new Aho();
    }

    public void insert(String s, int ind) {
        Aho pCrawl = this.ROOT;

        for(int i=0; i < s.length(); ++i) {
            int idx = s.charAt(i)-'a';

            if(pCrawl.CHILDREN[idx] == null)
                pCrawl.CHILDREN[idx] = new Aho();
            
            pCrawl = pCrawl.CHILDREN[idx];
        }

        pCrawl.PATTERN_INDEX = Math.min(pCrawl.PATTERN_INDEX, ind);
    }

    public boolean not_root(Aho pCrawl) {
        if(pCrawl == null || pCrawl == this.ROOT)
            return false;
        return true;
    }

    public void set_links() {

        this.ROOT.SUFFIX_LINK = this.ROOT;
        Aho temp, pCrawl;

        Queue<Aho> q = new LinkedList<>();
        q.offer(this.ROOT);

        while(!q.isEmpty()) {
            pCrawl = q.poll();
            
            for(int i=0; i<26; ++i) {

                if(not_root(pCrawl.CHILDREN[i])) {
                    q.offer(pCrawl.CHILDREN[i]);
                    temp = pCrawl.SUFFIX_LINK;

                    while(not_root(temp) && !not_root(temp.CHILDREN[i]))
                        temp = temp.SUFFIX_LINK;

                    if(not_root(pCrawl.CHILDREN[i]) && temp != pCrawl) {
                        pCrawl.CHILDREN[i].SUFFIX_LINK = temp.CHILDREN[i];
                        // pCrawl.CHILDREN[i].PATTERN_INDEX = Math.min(pCrawl.CHILDREN[i].PATTERN_INDEX, temp.CHILDREN[i].PATTERN_INDEX);
                    }
                    else pCrawl.CHILDREN[i].SUFFIX_LINK = this.ROOT;
                }
            }

            if(pCrawl.SUFFIX_LINK.PATTERN_INDEX < Integer.MAX_VALUE)
                pCrawl.OUTPUT_LINK = pCrawl.SUFFIX_LINK;
            else pCrawl.OUTPUT_LINK = pCrawl.SUFFIX_LINK.OUTPUT_LINK;
        }
    }

    public boolean search(String s, int ind) {
        Aho pCrawl = this.ROOT;

        for(int i=0; i < s.length(); ++i) {
            int idx = s.charAt(i)-'a';

            if(not_root(pCrawl.CHILDREN[idx])) {
                pCrawl = pCrawl.CHILDREN[idx];

                if(pCrawl.PATTERN_INDEX < ind)
                    return true;

                Aho mol = pCrawl.OUTPUT_LINK;
                while(not_root(mol)) {
                    if(mol.PATTERN_INDEX < index)
                        return true;
                    mol = mol.OUTPUT_LINK;
                }
            }
            else {
                while(pCrawl != this.ROOT && pCrawl.CHILDREN[idx] == null)
                    pCrawl = pCrawl.SUFFIX_LINK;

                if(pCrawl.CHILDREN[idx] != null) --i;
            }
        }

        return false;
    }
}

class SUffixArray {

    public int N;
    public String S;
    public int MAXN = 100010;
    public int[] LCP, RA, SA, TRA, TSA;

    public SUffixArray(String s) {
        this.S = s+"$";
        this.N = s.length();
        this.RA = new int[N]; this.TRA = new int[N];
        this.SA = new int[N]; this.TSA = new int[N];
    }

    public void cSort(int k) {
        int maxi = Math.max(300, N), sum = 0;

        int[] c = new int[MAXN];
        for(int i=0; i < N; ++i) ++c[(i+k < N ? RA[i+k] : 0)];

        for(int i=0; i < maxi; ++i) {
            int tp = c[i];
            c[i] = sum;
            sum += tp;
        }

        for(int i=0; i < N; ++i)
            TSA[c[(SA[i]+k > 0 ? RA[SA[i]+k] : 0)]++] = SA[i];

        for(int i=0; i < N; ++i) SA[i] = TSA[i];
    }

    public void build() {
        for(int k=1; k < N; k*=2) {
            cSort(k); cSort(0);
            
            int r = 0;
            TRA[SA[0]] = r;

            for(int i=1; i < N; ++i) {
                if(RA[SA[i]] == RA[SA[i-1]] && RA[SA[i]+k] == RA[SA[i-1]+k]) TRA[SA[i]] = r;
                else TRA[SA[i]] = ++r;
            }

            for(int i=0; i < N; ++i) RA[i] = TRA[i];
            if(RA[SA[N-1]] == N-1) break;
        }
    }

    public void kasai() {
        int k = 0;
        this.LCP = new int[N];
        int[] rank = new int[N];

        for(int i=0; i < N; ++i) rank[SA[i]] = i;

        for(int i=0; i < N; ++i) {
            if(rank[i] == N-1) {
                k = 0;
                continue; }

            int j = SA[rank[i]+1];
            while(i+k < N && j+k < N && S.charAt(i+k) == S.charAt(j+k))
                ++k;

            LCP[rank[i]] = k;
        }
    }
}
