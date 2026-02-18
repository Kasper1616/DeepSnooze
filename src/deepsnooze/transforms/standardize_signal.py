class StandardizeSignal:
    def __call__(self, signal):
        # signal: [3, 512]
        return (signal - signal.mean()) / (signal.std() + 1e-7)