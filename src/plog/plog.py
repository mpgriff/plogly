from matplotlib.cm import Blues
import matplotlib
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os


class abcLog(ABC):
    def __init__(self, values, depth_top, depth_bot, log_name, units='', xy=(0, 0)):
        assert len(depth_bot) == len(
            depth_top), "top and bottom depth axis must have the same length"
        assert len(depth_bot) == len(
            values),     "log valuess and depth axes must have the same length"
        self.depth_top = np.array(depth_top)
        self.depth_bot = np.array(depth_bot)
        self.values = np.array(values)
        self.name = log_name
        self.units = units
        self.x, self.y = xy
        self.elev = None

    @abstractmethod
    def plot(self, ax=None, **kwargs):
        pass

    @property
    def z(self):
        return 0.5*(self.depth_top + self.depth_bot)

    @property
    def thickness(self):
        return np.abs(self.depth_bot-self.depth_top)

    @property
    def Nz(self):
        return len(self.values)

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def label_values(self, ax, xpos=1):
        for val, y_pos in zip(self.values, self.z):
            ax.text(xpos, y_pos, str(val), verticalalignment='center')

    def elevation(self, elev):
        if self.elev is not None:
            raise valuesError("elevation cannot be re-assigned")

        self.elev = elev
        for mem in vars(self).keys():
            if mem.startswith('depth'):
                exec_str = f'self.{mem} = elev - self.{mem}'
                exec(exec_str)
        return self

    @classmethod
    def load(cls, fname, **kwargs):
        """load saved log object

        Args:
            fname (str): path to model file

        Returns:
            Model object

        """
        from dill import load
        a_file = open(fname, "rb")
        akern = load(a_file)
        a_file.close()
        return akern

    def save(self, fname):
        """ Save model object

        Args:
            fname (str): path to desired save location
        Returns:
            None
        """
        from dill import dump
        outfile = open(fname, 'wb')
        dump(self, outfile)
        outfile.close()


class Log(abcLog):
    def __init__(self, values, depth_top, depth_bot, log_name, units='', cmap=Blues, **kwargs):
        super().__init__(values, depth_top, depth_bot, log_name, units=units, **kwargs)

        if isinstance(self.values[0], int) or isinstance(self.values[0], float):
            norm_val = self.values / np.abs(self.values).max()
            if np.any(norm_val < 0.):
                norm_val = 0.5 + 0.5*norm_val
            self.color = {val: cmap(nval)
                          for val, nval in zip(self.values, norm_val)}

    def plot(self, ax=None, x_offset=0., **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        cbar = False
        if 'cmap' in kwargs:
            cmap = kwargs.pop('cmap')
            dx = kwargs.pop('dx', 1.)
            border = kwargs.pop('border', '')

            if isinstance(dx, float) or isinstance(dx, int):
                dx = np.ones(self.Nz)*dx
            cbar = True

            X = np.outer(np.array([-0.25, 0.25]), dx)
            Z = np.repeat(self.z, 2).reshape(self.Nz, 2).T
            V = np.repeat(self.values, 2).reshape(self.Nz, 2).T
            pcm = ax.pcolor(X+x_offset, Z, V, cmap=cmap, **kwargs)
            tmpz = Z

            if border != '':
                ax.plot(X[0]*2+x_offset, Z[0], border)
                ax.plot(X[1]*2+x_offset, Z[1], border)

        elif np.all(self.depth_bot == self.depth_top):
            ax.plot(self.values+x_offset, self.z, **kwargs)
            tmpz = self.z
        else:
            vmin, vmax = kwargs.pop('vmin', self.values.min()), kwargs.pop(
                'vmax', self.values.max())
            # this gives the plot a step-wise form, where the layer boundaries are correct.
            tmpz = np.vstack((self.depth_top, self.depth_bot)).T.flatten()
            tmpx = np.vstack((self.values, self.values)).T.flatten()
            ax.plot(tmpx+x_offset, tmpz, **kwargs)
            kwargs['vmin'], kwargs['vmax'] = vmin, vmax
        if self.units != '':
            ax.set_xlabel(f'[{self.units}]')

        if self.elev is None:
            ax.set_ylim(tmpz.max(), tmpz.min())

        if cbar:
            res = (ax, pcm)
        else:
            res = ax

        return res

    def plot_cyklo(self, ax=None, elevation=0., xy=(0, 0), unit_rad=1., dr=0.2, m_per_turn=100.):
        dr *= unit_rad
        if ax is None:
            fig, ax = plt.subplots()

        # The inner ring starts at "9 o'clock" and goes clockwise upwards from elevation 0.
        # On the outside, a new ring is placed from "9 o'clock" every 100 meters downwards clockwise.
        elev_top = elevation-self.depth_top
        elev_bot = elevation-self.depth_bot

        angle_top = np.pi - 2*np.pi*elev_top/m_per_turn
        angle_bot = np.pi - 2*np.pi*elev_bot/m_per_turn

        for i, (phi_t, phi_b, geo) in enumerate(zip(angle_top, angle_bot, self.values)):

            arc = np.linspace(phi_t, phi_b, int(
                1000.*np.abs(phi_b-phi_t)/(2.*np.pi)))
            r_mid = unit_rad - 1.25*dr*(arc-np.pi)/(2.*np.pi)

            x1 = (r_mid-dr/2.)*np.cos(arc)
            y1 = (r_mid-dr/2.)*np.sin(arc)

            x2 = (r_mid+dr/2.)*np.cos(arc)
            y2 = (r_mid+dr/2.)*np.sin(arc)

            x = np.concatenate((x1, x2[::-1]))
            y = np.concatenate((y1, y2[::-1]))
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            poly = Polygon(
                np.stack((x+xy[0], y+xy[1])).T, facecolor=self.color[geo])
            ax.add_patch(poly)
        ax.plot([-1*unit_rad+xy[0]], [0*unit_rad+xy[1]], 'w_')
        ax.plot(xy[0], xy[1], 'w.')
        return ax

    def sample(self, z):
        i_top = np.clip(np.searchsorted(
            self.depth_top, z, side='right'), 0, self.Nz-1)
        i_bot = np.clip(np.searchsorted(
            self.depth_bot, z, side='left'),  0, self.Nz-1)
        val_top = self.values[i_top]
        val_bot = self.values[i_bot]
        # val_top[val_top!=val_bot] = np.nan
        return val_bot

    def rediscretize(self, ref_log):
        """this function rediscretizes valuess onto a new grid. if an interval on the new discretization spans several layers, the resulting average will be thickness averaged."""

        if self.elev is None:
            z_old1, z_old2 = self.depth_top, self.depth_bot
        else:
            z_old1, z_old2 = self.depth_bot, self.depth_top

        if ref_log.elev is None:
            z_new1, z_new2 = ref_log.depth_top, ref_log.depth_bot
        else:
            z_new1, z_new2 = ref_log.depth_bot, ref_log.depth_top

        zo1, zn1 = np.meshgrid(z_old1, z_new1)
        zo2, zn2 = np.meshgrid(z_old2, z_new2)

        ovlp1 = np.max(np.stack([zo1, zn1]), axis=0)
        ovlp2 = np.min(np.stack([zo2, zn2]), axis=0)
        ovlp = ovlp2-ovlp1
        ovlp[ovlp < 0] = 0.
        norm = ovlp.sum(1, keepdims=True)
        norm[norm == 0] = np.nan
        ovlp /= norm

        new_log = ref_log.copy()
        new_log.values = ovlp@self.values
        return new_log

    @classmethod
    def standard(cls, values, depth, name, **kwargs):
        """assumed to start at the surface, and extend to infinity in the bottom layer"""
        depth = np.array(depth)
        depth_top = np.insert(depth, 0, 0)
        depth_bot = np.append(depth, 1.1*depth.max())
        return cls(values, depth_top, depth_bot, name, **kwargs)

    @classmethod
    def geology(cls, geology, depth_top, depth_bottom, name='geology', color_dictionary=None, **kwargs):
        self = cls(geology, depth_top, depth_bottom, name, **kwargs)

        if color_dictionary is None:
            import matplotlib.colors as mcolors
            self.color = {x: c for x, c in zip(
                np.unique(np.array(self.values)), mcolors.TABLEAU_COLORS.keys())}
        elif isinstance(color_dictionary, str):
            from matplotlib import colormaps
            uniq_geo = np.unique(np.array(self.values))
            colors = colormaps[color_dictionary](
                np.linspace(0., 1., len(uniq_geo)))
            self.color = {x: c for x, c in zip(uniq_geo, colors)}
        else:
            self.color = color_dictionary

        def plot(ax=None, dx=1., hatch=None, label=True):
            if ax is None:
                fig, ax = plt.subplots()
            if hatch is None:
                hatch = [None]*len(self.values)

            for i, (dpth_t, dpth_b, geo) in enumerate(zip(self.depth_top, self.depth_bot, self.values)):
                layer = np.array([[-dx/2, dpth_t],     [dx/2, dpth_t],
                                  [dx/2, dpth_b],  [-dx/2, dpth_b],
                                  [-dx/2, dpth_t]])
                poly = Polygon(
                    layer, facecolor=self.color[geo], hatch=hatch[i])
                ax.add_patch(poly)
            ax.set_xlim([-dx/2, dx/2])
            ax.set_ylim(max(self.z)+max(self.thickness), 0)
            ax.set_xticks([])
            # ax.set_aspect(0.5)

            if label:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                axgeo = ax.secondary_yaxis('left')
                axgeo.set_yticks(self.z)
                axgeo.set_yticklabels(self.values, rotation=45)
            ax.set_ylabel('depth [m]')
            return ax
        self.plot = plot
        return self

    @classmethod
    def two_dim(cls, depth, x_axis, values, name, **kwargs):
        assert len(values.shape) == 2, "data is not 2d, try using plain Log class"
        self = cls(values, depth, depth, name, **kwargs)
        self.x_axis = x_axis

        def plot(ax=None, **kwargs):
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            ax.pcolor(self.x_axis, self.z, self.values, **kwargs)
            return ax

        self.plot = plot
        return self


class Borehole: 
    def __init__(self, logs, elevation=0., name='', x=None, y=None):
        self.name = name
        self.logs = logs
        self.elev = elevation

        if x is not None:
            for lg in self.logs:
                lg.x = x
        if y is not None:
            for lg in self.logs:
                lg.y = y

    def elev2depth(self, elev):
        return self.elev-elev

    def depth2elev(self, depth):
        return self.elev-depth

    def elevation(self):
        for logs in self.logs:
            logs.elevation(self.elev)

    def __iter__(self):
        return iter(self.logs)

    @property
    def names(self):
        return [x.name for x in self]

    @property
    def x(self):
        return mean([lg.x for lg in self])

    @property
    def y(self):
        return mean([lg.y for lg in self])

    def __getitem__(self, logname):
        i = self.names.index(logname)
        tmp_log = self.logs[i].copy()

        return tmp_log

    def plot(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1, len(self.logs), sharey=True)
        else:
            fig = axs.figure

        if isinstance(axs, matplotlib.axes._axes.Axes):
            axs = np.array([axs])

        for i, log in enumerate(self.logs):
            log.plot(ax=axs[i])
            axs[i].set_title(log.name)
            if self.elev != 0:
                ax2 = axs[i].secondary_yaxis(
                    'right', functions=(self.elev2depth, self.depth2elev))
            if log.name=='geology':
                axs[i].set_aspect(0.15)
        return axs

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    @classmethod
    def load(cls, fname, **kwargs):
        """load saved borehole object

        Args:
            fname (str): path to model file

        Returns:
            Borehole object

        """
        from dill import load
        a_file = open(fname, "rb")
        akern = load(a_file)
        a_file.close()
        return akern

    def save(self, fname):
        """ Save model object

        Args:
            fname (str): path to desired save location
        Returns:
            None
        """
        from dill import dump
        outfile = open(fname, 'wb')
        dump(self, outfile)
        outfile.close()

class Dart(Borehole):
    def __init__(self, export_folder, **kwargs):
        logs = []

        bh_name = os.path.split(export_folder)[-1]
        # bh_name = ''
        raw = np.genfromtxt(export_folder+'_1Dvectors.txt', names=True)
        for name in raw.dtype.names:
            if name not in ['depth', 'unix_time', 'board_temp', 'magnet_temp']:
                tmp_log = Log(raw[name], raw['depth']-0.22 *
                              0.5, raw['depth']+0.22*0.5, name)
                logs.append(tmp_log)

        SE_decay = np.genfromtxt(export_folder+'_SE_decay.txt')
        SE_time = np.genfromtxt(export_folder+'_SE_decay_time.txt')
        # bit of a hack
        logs.append(Log.two_dim(logs[-1].z, SE_time,
                    SE_decay[:, :-1], 'SE decay'))

        T2_dist = np.genfromtxt(export_folder+'_T2_dist.txt')
        T2_dist_bins = 10**np.genfromtxt(export_folder+'_T2_bins_log10s.txt')
        # bit of a hack
        logs.append(Log.two_dim(
            logs[-1].z, T2_dist_bins, T2_dist[:, 1:], 'T2 dist'))


        super().__init__(logs, **kwargs)
        self.export_folder = export_folder

    def plot_wc(self, ax=None, legend=False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        base = np.zeros_like(self['freef'].values)
        ax.plot(self['totalf'].values,
                self['totalf'].z,  'k-', label='total',)
        ax.fill_betweenx(
            self['totalf'].z, base, self['freef'].values, label='free', facecolor='b')
        base += self['freef'].values
        ax.fill_betweenx(self['totalf'].z, base, base +
                            self['capf'].values, label='cap.', facecolor='cyan')
        base += self['capf'].values
        ax.fill_betweenx(self['totalf'].z, base, base +
                            self['clayf'].values, label='clay', facecolor='bisque')
        if legend:
            ax.legend(fontsize='x-small')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Water Content [%]')
        return ax

    def plot(self, axs=None):
        n_extra = len(self.logs)-15
        if axs is None:

            width_ratios = [1]*n_extra + [1, 2, 3, 1, 1]
            fig, axs = plt.subplots(
                1, n_extra+5, sharey=True, width_ratios=width_ratios, figsize=(8,4))
        else:
            assert len(axs.flatten(
            )) >= 6, "not enough subplots provided for a dart logging data display"
            fig = axs.flatten()[0].figure

        for i in range(n_extra):
            self.logs[i].plot(ax=axs[i])
        self.plot_wc(ax=axs[n_extra])
        
        self['SE decay'].plot(ax=axs[n_extra+1])
        axs[n_extra+1].set_xlabel('SE decay [s]')
        
        self['T2 dist'].plot(ax=axs[n_extra+2], cmap='Greens');
        self['mlT2'].plot(ax=axs[n_extra+2], color='r')
        axs[n_extra+2].set_xlabel('T2 dist [s]')
        axs[n_extra+2].set_xscale('log')

        tmp_logs = self.names
        tmp_logs.pop(tmp_logs.index('T2 dist'))
        tmp_logs.pop(tmp_logs.index('SE decay'))
        tmp_logs.pop(tmp_logs.index('noise'))

        for param in ['Ksdr', 'Ktc', 'Ksoe']:
            axs[n_extra+3].semilogx(self[param].values,
                                    self[param].z, drawstyle='steps-mid', label=param)
        axs[n_extra+3].set_xlabel('K [m/day]')

        # for param in ['Tsdr', 'Ttc', 'Tsoe']:
        #     axs[n_extra+4].semilogx(self[param].values, self[param].z,
        #                                 drawstyle='steps-mid', label=param)
        # axs[n_extra+4].plot(self['soe'].values, self['soe'].z,
        #                     drawstyle='steps-mid', label='soe')
        # axs[n_extra+4].set_xlabel('T [m$^2$/day]')

        axs[n_extra+3].legend(fontsize='x-small')
        axs[n_extra+4].legend(fontsize='x-small')
        self['noise'].plot(ax=axs[n_extra+5])
        axs[n_extra+5].set_xlabel('noise [%]')
        axs[n_extra+0].set_ylim(self['totalf'].z.max(),
                                self['totalf'].z.min())
        axs[n_extra+5].yaxis.set_label_position("right")
        axs[n_extra+5].yaxis.tick_right()
        axs[n_extra+5].yaxis.set_label('depth [m]')
        return axs

    def t2_trim(self, T2_min):
        # new_bh = self.copy()
        tmp = self['T2 dist'].copy()
        five_ms = tmp.x_axis >= T2_min
        lg1 = Log.two_dim(
            tmp.z, tmp.x_axis[five_ms], tmp.values[:, five_ms], 'T2 dist')
        mlT2 = 10**(np.sum(np.log10(lg1.x_axis[None, :])
                    * lg1.values, axis=1) / np.sum(lg1.values, axis=1))
        lg2 = Log(mlT2, tmp.z, tmp.z, 'mlT2')
        lg3 = Log(np.sum(lg1.values, axis=1), tmp.z, tmp.z, 'totalf')
        self.logs[self.names.index('T2 dist')] = lg1.copy()
        self.logs[self.names.index('mlT2')] = lg2.copy()
        self.logs[self.names.index('totalf')] = lg3.copy()



def proj_func_maker(xy1, xy2):
    (x1, y1) = xy1
    (x2, y2) = xy2
    m = (y2 - y1) / (x2 - x1)
    norm = np.sqrt(1. + m ** 2)
    ux, uy = 1.0 / norm, m / norm
    mx_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    if x2 < x1:
        ux *= -1.
        uy *= -1.

    def dist_func(x, y):
        dist = (x - x1) * ux + (y - y1) * uy
        if dist < 0.:
            dist = np.inf
        return dist

    def dist_func_inv(dist):
        return (dist * ux + x1, dist * uy + y1)

    return dist_func, dist_func_inv


def piecewise_axis(pnts):
    x = np.array([pnt[0] for pnt in pnts])
    y = np.array([pnt[1] for pnt in pnts])
    dx = np.insert(np.diff(x), 0, 0)
    dy = np.insert(np.diff(y), 0, 0)

    seg_dist = np.sqrt(dx ** 2 + dy ** 2)
    tot_dist = np.cumsum(seg_dist)

    dist_funcs = []
    inv_dist_funcs = []
    for i in range(len(seg_dist)):
        if i + 1 < len(seg_dist):
            f, invf = proj_func_maker(pnts[i], pnts[i + 1])
            dist_funcs.append(f)
            inv_dist_funcs.append(invf)

    def dist_func(x, y, return_perp=False):
        min_dist = np.inf
        min_index = None
        res = (np.inf, np.inf)
        for i, (f, invf) in enumerate(zip(dist_funcs, inv_dist_funcs)):
            cdist = f(x, y)
            px, py = invf(cdist)
            d_proj = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if d_proj < min_dist:
                min_dist = cdist
                res = (tot_dist[i]+cdist, d_proj)

        if not return_perp:
            res = res[0]
        return res

    def dist_func_inv(dist):
        i_dist = np.searchsorted(tot_dist, dist, side='left')-1
        if i_dist >= len(inv_dist_funcs):
            i_dist = -1
        f = inv_dist_funcs[i_dist]
        return f(dist - tot_dist[i_dist])

    return dist_func, dist_func_inv


class Section:
    def __init__(self, logs, proj_pnts=None, **kwargs):
        self.logs = logs

        self.min_val = min([lg.values.min() for lg in self])
        self.max_val = max([lg.values.max() for lg in self])
        self.x0 = kwargs.get('x0', self.x.min())
        self.y0 = kwargs.get('y0', self.y[np.argmin(self.x)])
        self.name = kwargs.get('name', self[0].name)
        if proj_pnts is None:
            proj_pnts = [(lg.x, lg.y) for lg in self]
            while len(proj_pnts) > 500:
                proj_pnts = proj_pnts[::2]
        self.make_projection_axis(*proj_pnts)

    def __iter__(self):
        return iter(self.logs)

    def make_projection_axis(self, *points):
        assert len(points) > 1, "need atleast two points to define the axis"
        if len(points) == 2:
            self.dist_func, self.dist_func_inv = proj_func_maker(*points)
        else:
            self.dist_func, self.dist_func_inv = piecewise_axis(points)
        idx_sort = np.argsort(self.x_dist)
        log_sort = []
        for i in idx_sort:
            log_sort.append(self[i])
        self.logs = log_sort

    @property
    def units(self):
        return self[0].units

    @property
    def x(self):
        return np.array([bh.x for bh in self])

    @property
    def y(self):
        return np.array([bh.y for bh in self])

    @property
    def x_dist(self):
        return np.array([self.dist_func(x, y) for x, y in zip(self.x, self.y)])

    @property
    def proj_points(self):
        dist = self.x_dist
        x, y = np.zeros_like(dist), np.zeros_like(dist)
        for i, d in enumerate(dist):
            x[i], y[i] = self.dist_func_inv(d)
        return x, y

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def __getitem__(self, i):
        return self.logs[i]

    def plot(self, axs=None, **kwargs):
        offset = kwargs.get('offset', False)

        kwargs.setdefault('vmin', self.min_val)
        kwargs.setdefault('vmax', self.max_val)

        if axs is None:
            fig, axs1 = plt.subplots(
                1, len(self.logs), sharex=True, sharey=True)
        elif isinstance(axs, plt.Axes):
            axs1 = [axs]*len(self.logs)
            offset = True
        else:
            assert len(axs) == len(
                self.logs), "provided axs doesnt match the number logs"
            axs1 = axs

        xbnds = [0, 0]
        dx = kwargs.get('dx', 1.)
        x_slide = kwargs.pop('x_offset', 0.)
        if not (isinstance(dx, float) or isinstance(dx, int)):
            dx = max(dx)
        for ax, log in zip(axs1, self):
            x_offset = self.dist_func(log.x, log.y) if offset else 0
            x_offset += x_slide
            xbnds[0] = min(x_offset-dx*2, xbnds[0])
            xbnds[1] = max(x_offset+dx*2, xbnds[1])
            ax = log.plot(ax=ax, x_offset=x_offset, **kwargs)

        if offset:
            axs1[0].set_xlim(xbnds)

        if not isinstance(ax, plt.Axes):
            plt.colorbar(ax[1], ax=ax[0], fraction=0.01)
        return axs1

    def plot_interpolated(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        X, Z, Val = self.interpolate_xsection(
            max_bin_skip=kwargs.pop('max_bin_skip', 1))
        cmap = kwargs.pop('cmap', 'rainbow' if self.name ==
                          'WC' else 'seismic')
        if 'levels' in kwargs or 'contour_lines' in kwargs:
            pcm = ax.contourf(X, Z, Val, cmap=cmap, **kwargs)
            if kwargs.get('contour_lines', False):
                pcm2 = ax.contour(pcm, levels=pcm.levels,
                                  linewidth=0.1, colors='k')
                ax.clabel(pcm2, fmt='%2.0f', colors='k', fontsize=11)
        else:
            pcm = ax.pcolor(X, Z, Val, cmap=cmap, shading='auto', **kwargs)

        if kwargs.get('cbar', True):
            label = self.name
            if not self.units == '':
                label += f' [{self.units}]'
            plt.colorbar(pcm, ax=ax, fraction=0.01, cmap=cmap, label=label)
        return ax

    def interpolate_xsection(self, max_bin_skip=1):
        if len(self.logs) < 20:
            from scipy.interpolate import interp2d, griddata
            x_dist = self.x_dist
            DX = []
            DZ = []
            Dpmtr = []

            for i, log in enumerate(self):
                DX.append(x_dist[i]*np.ones_like(log.z))
                DZ.append(log.z)
                Dpmtr.append(log.values)

            DX = np.concatenate(DX)
            DZ = np.concatenate(DZ)
            Dpmtr = np.concatenate(Dpmtr)
            # f_wc = interp2d(DX,DZ, Dpmtr, kind='linear')
            X, Z = np.meshgrid(np.linspace(x_dist.min(), x_dist.max(), len(
                self.logs)*10), np.linspace(DZ.min(), DZ.max(), log.Nz*10))

            iWC = griddata((DX, DZ), Dpmtr, (X, Z))
            return X, Z, iWC
        else:
            from scipy.interpolate import interp1d
            x_dist = self.x_dist
            xp, yp = self.proj_points
            perp_dist = np.sqrt((self.x-xp)**2 + (self.y-yp)**2)

            avg_dx = (x_dist.max() / len(self.logs))*1.05
            x_mesh = 0.5 + np.arange(len(self.logs)+2)*avg_dx

            all_z = np.unique(np.concatenate([lg.z for lg in self]))
            xsec = np.zeros(x_mesh.shape+all_z.shape)
            n = np.zeros_like(x_mesh)
            for i, lg in enumerate(self):
                f = interp1d(lg.z, lg.values, bounds_error=False,
                             fill_values=np.nan)
                w = 1./perp_dist[i]
                idx = np.searchsorted(x_mesh, x_dist[i])
                xsec[idx] += f(all_z)*w
                n[idx] += w

            xsec /= n[:, None]
            xsec[n == 0, :] = np.nan
            X, Z = np.meshgrid(x_mesh, all_z, indexing='ij')

            nan_wall = np.isnan(xsec)

            # if an unfilled bin has data on either side; skip it and let the cmap interpolate over it instead of leaving it as a nan.
            nan_count = 0
            nan_count_v = np.zeros(len(xsec[:, 0]))
            i_st = 0
            for i, v in enumerate(np.all(nan_wall, axis=-1)):
                if v:
                    if nan_count == 0:
                        i_st = i
                    nan_count += 1
                elif nan_count > 0:
                    nan_count_v[i_st:i] = nan_count
                    nan_count = 0
            keep = nan_count_v > max_bin_skip
            keep[nan_count_v == 0] = True
            # keep = nan_count_v != 1
            X = X[keep, :]
            Z = Z[keep, :]
            xsec = xsec[keep, :]

            return X, Z, xsec