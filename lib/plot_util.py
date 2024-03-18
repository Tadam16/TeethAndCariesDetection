from path_util import out_dir

fig_num = 0


def save_next(fig, name, with_fig_num=True):
    """Simple function to save a figure to the out_dir with a name and an optional sequential figure number"""
    global fig_num
    out_dir.mkdir(exist_ok=True)
    if with_fig_num:
        fig.savefig(out_dir / f'{name}_{fig_num}.png')
        fig_num += 1
    else:
        fig.savefig(out_dir / f'{name}.png')
