def get_class_weights(cw):
    if cw == 'no1':
        class_weights = [0.05, 0.2, 0.8, 0.7, 0.4]
    elif cw == 'distr':
        class_weights = [1 / 0.030342701529957234, 1 / 0.023289585196743044, 1 / 0.002574037714986485, 1 / 0.002682490082519425, 1 / 0.0017965885357082826]
        sum_of_weights = sum(class_weights)
        class_weights = [w / sum_of_weights for w in class_weights]
    elif cw == 'distr_no_overlap':
        class_weights = [32.20398121025673, 41.516691841904844, 406.2242790072747, 319.5142994620793, 727.8449005124751]
        sum_of_weights = sum(class_weights)
        class_weights = [w / sum_of_weights for w in class_weights]
    elif cw == 'equal':
        class_weights = [0.2] * 5
    else:
        raise ValueError(f'Not implemented for class weight choice: {cw}')
    return class_weights
