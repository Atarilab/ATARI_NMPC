from ..core.pinocchio_fwd import *
import numpy as np


def loadModelImpl(
    model_path: str, pkg_path: str = "rsc/", loadGeom: bool = False
) -> (
    tuple[pinocchio.Model, pinocchio.GeometryModel, pinocchio.GeometryModel]
    | pinocchio.Model
):
    root = pinocchio.JointModelComposite(2)
    root.addJoint(pinocchio.JointModelTranslation())
    root.addJoint(pinocchio.JointModelSphericalZYX())
    if loadGeom:
        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
            model_path, root_joint=root, package_dirs=pkg_path
        )
    else:
        model = pinocchio.buildModelFromUrdf(model_path, root_joint=root)
    model.lowerPositionLimit[:3] = np.array([-1e11] * 3)
    model.upperPositionLimit[:3] = np.array([1e11] * 3)
    model.lowerPositionLimit[3:6] = np.array([-3.14] * 3)
    model.upperPositionLimit[3:6] = np.array([3.14] * 3)
    model.velocityLimit[:6] = 1e11
    if loadGeom:
        return model, collision_model, visual_model
    else:
        return model


def loadModelGeom(
    model_path: str, **kwargs
) -> tuple[pinocchio.Model, pinocchio.GeometryModel, pinocchio.GeometryModel]:
    return loadModelImpl(model_path, loadGeom=True, **kwargs)


def loadModel(model_path: str, **kwargs) -> tuple[pinocchio.Model, pinocchio.Data]:
    model = loadModelImpl(model_path, **kwargs)
    data = model.createData()
    return model, data


def toSymModel(model: pinocchio.Model) -> tuple[pin.Model, pin.Data]:
    model_ = pin.Model(model)
    data_ = model_.createData()
    return model_, data_


def loadSymModel(model_path: str, **kwargs) -> tuple[pin.Model, pin.Data]:
    return toSymModel(loadModelImpl(model_path, **kwargs))
