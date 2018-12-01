import torch
import numpy as np
import timeit


class SnakePytorch:
    """
    class that implement snake algorithm on GPU
    may run 4-5 times faster than multicores algorithm on CPU
    need to know beforehand the batch size and N, M
    """

    def __init__(self, delta, b_sz, N, M):
        self.delta = delta
        self.b_sz = b_sz
        self.N = N
        self.M = M
        # allocate big arrays to save allocation time later
        self.E = 1e10 * torch.ones((b_sz, N, M + 2 * delta, M + 2 * delta)).cuda()
        self.valueF = torch.zeros((b_sz, N - 1, M, M + 2 * delta)).cuda()
        self.indexF = torch.zeros((b_sz, N - 1, M, M)).cuda()
        self.ind = torch.zeros((b_sz, N), dtype=torch.long)

        # all points not satisfy smoothness constrain will receive a 1e10 penalty
        # for the enery function
        energy_mask = 1e10 * torch.ones(M + 2 * delta, M + 2 * delta).cuda()
        for i in range(delta, M + delta):
            for j in range(max(delta, i - delta), min(M + delta, i + delta + 1)):
                # no penalty for valid point
                energy_mask[i, j] = 0
        self.energy_mask = energy_mask

    def reset(self):
        self.valueF.fill_(1e10)

    def __call__(self, g):
        """
        :param g: tensor of g, size [..., batch_size, height, width]
        :param delta: int. smoothness constraint
        :return: 2d index of the radial lines, size [..., batch_size, num_lines, 1]
        """
        with torch.no_grad():
            b_sz = self.b_sz
            N = self.N
            M = self.M

            self.reset()
            E = self.E
            valueF = self.valueF
            indexF = self.indexF
            ind = self.ind
            delta = self.delta

            g_shape = g.shape
            g = g.reshape((b_sz, N, M))

            for n in range(N):
                l = (n + 1) % N
                E[:, n, delta:M + delta, delta:M + delta] = \
                    (g[:, n, :].reshape((-1, M, 1)) + g[:, l, :].reshape((-1, 1, M))) / 2

            # E is now shape [bs, N, M+2*delta, M+2*delta]
            # M -> M + 2*delta: map energy function from M+2*delta values on this line to next lines
            # for the next line, M+2*delta is extended energy matrix that includes infinite g values
            # for delta positions both before 0 and after M
            E = E.add_(self.energy_mask)

            # factor k out as well
            # this implementation tested to be correct. valueF match the original
            for n in range(N - 1):
                if n == 0:
                    # V1:
                    # contain energy values of M valid points on 1st line to
                    # M + 2 * delta points on 2nd lines,
                    # including the points outside range before 0 and after M
                    # V2:
                    # contain energy values of M + 2 * delta points on 2nd lines
                    # to M valid points on 3rd line
                    vs = E[:, 0, delta:M + delta, :].reshape((-1, M, M + 2 * delta, 1)) + \
                         E[:, 1, :, delta:M + delta].reshape((-1, 1, M + 2 * delta, M))
                else:
                    # similar to above
                    vs = valueF[:, n - 1, :, :].reshape((-1, M, M + 2 * delta, 1)) + \
                         E[:, n + 1, :, delta:M + delta].reshape((-1, 1, M + 2 * delta, M))
                # Now our vs will be v1 + v2
                vs = vs.permute((0, 1, 3, 2))
                valueF[:, n, :, delta:delta + M], indexF[:, n, ...] = torch.min(vs, dim=-1)
                # minus delta because valid points start from index delta
                indexF[:, n, ...] = indexF[:, n, ...] - delta

            # backtrack
            # initial benchmarking shows that the indexing line below take a very long time
            # almost equal to the runtime of the whole function
            # subsequent test by placeing a print function of vs above every iteration
            # reveal that it's working correct. pytorch might have some kind of "non-blocking"
            # that allow the code below to run before finishing the computation code above.
            # However, the indexing operation requires the result above anyway.
            r_M = torch.cuda.LongTensor(np.asarray(list(range(M))))
            vs = valueF[:, N - 2, r_M, r_M + delta]
            ind[:, 0] = torch.argmin(vs, dim=-1)
            ind[:, N - 1] = indexF[range(b_sz), N - 2, ind[:, 0], ind[:, 0]]
            for n in range(N - 2, -1, -1):
                ind[:, n] = indexF[range(b_sz), n-1, ind[:, 0], ind[:, n + 1]]

            ind = ind.reshape(list(g_shape[:-2]) + [N, 1]).clone()

            return ind


if __name__ == "__main__":
    # seed_all(0)
    # before changes: 2.1s for b_sz=100, N=M=80
    # after factorize k: 0.46s
    # test correct valueF using this configuration: bsz=1, N=M=5
    # value of test see in vs_values.txt
    b_sz = 100
    N = 60
    M = 60
    delta = 2
    snake = SnakePytorch(delta, b_sz, N, M)
    g_np = np.random.random((b_sz, 1, N, M))
    g_cuda = torch.from_numpy(g_np).type(torch.cuda.FloatTensor)
    for _ in range(20):
        start = timeit.default_timer()
        ind = snake(g_cuda)
        stop = timeit.default_timer()
        print("total_time", stop - start)
