class VirtualSingleTargetAction:
    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return '<{} [{}]>'.format(self.__class__.__name__, self.target)


class VirtualMultiTargetAction:
    def __init__(self, targets):
        self.targets = targets

    def __repr__(self):
        return '<{} {}>'.format(
                self.__class__.__name__,
                ', '.join('[{}]'.format(t) for t in self.targets))


class TrackAction(VirtualSingleTargetAction):
    pass


class ResolveLinkAction(VirtualSingleTargetAction):
    def __init__(self, target):
        super().__init__(target)
        self.resolve_to = os.readlink(target)


class UntrackAction(VirtualSingleTargetAction):
    pass


class DeleteAction(VirtualSingleTargetAction):
    pass


class GlobAction(VirtualSingleTargetAction):
    pass


class GlobAllAction(VirtualSingleTargetAction):
    pass


class RenameAction(VirtualMultiTargetAction):
    def __init__(self, src, dst):
        super().__init__((src, dst))

    @property
    def src(self):
        return self.targets[0]

    @property
    def dst(self):
        return self.targets[1]


class DominoRenameAction(VirtualMultiTargetAction):
    pass


class RotateRenameAction(VirtualMultiTargetAction):
    pass
