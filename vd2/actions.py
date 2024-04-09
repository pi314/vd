class VirtualAction:
    def __init__(self, *targets):
        self.targets = targets

    @property
    def src(self):
        return self.targets[0]

    @property
    def dst(self):
        return self.targets[1]

    def __repr__(self):
        return '<{} {}>'.format(
                self.__class__.__name__,
                ', '.join('[{}]'.format(t) for t in self.targets))


class MetaAction(VirtualAction):
    pass


class FSAction(VirtualAction):
    pass


class TrackAction(MetaAction):
    pass


class ResolveLinkAction(MetaAction):
    pass
    # os.readlink(target)


class UntrackAction(MetaAction):
    pass


class GlobAction(MetaAction):
    pass


class GlobAllAction(MetaAction):
    pass


class DeleteAction(FSAction):
    pass


class RenameAction(FSAction):
    pass


class CopyAction(FSAction):
    pass


class DominoRenameAction(FSAction):
    pass


class RotateRenameAction(FSAction):
    pass
