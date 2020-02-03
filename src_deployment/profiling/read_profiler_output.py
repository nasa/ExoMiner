import pstats

def read_profiler_output(filepath):
    p = pstats.Stats(filepath)
    p.strip_dirs().sort_stats(-1).print_stats()

if __name__ == '__main__':

    read_profiler_output('profile.txt')