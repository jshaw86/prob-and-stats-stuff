

def longest_common_substring(s1, s2):
    m = len(s1)
    n = len(s2)
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                lcs[i + 1][j + 1] = lcs[i][j] + 1
                if lcs[i + 1][j + 1] > longest:
                    lcs_set = set()
                    longest = lcs[i + 1][j + 1]
                    lcs_set.add(s1[i - longest + 1:i + 1])
                elif lcs[i + 1][j + 1] == longest:
                    lcs_set.add(s1[i - longest + 1:i + 1])

    return lcs_set
