from multiprocessing import Process, Queue


def foo(i):
    print(i)
    return i


if __name__ == "__main__":
    import sys

    qin = Queue()
    qout = Queue()
    worker = Process(target=foo, args=[0])
    worker.start()
    out = worker.join()
    print(out)
