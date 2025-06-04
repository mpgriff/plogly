from matplotlib.cm import Blues
import matplotlib
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from matplotlib import ticker


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
        if len(self.values.shape)==1:
            new_log = self.__class__(self.values, self.depth_top, self.depth_bot, self.name, units=self.units, xy=(self.x, self.y))

        
        elif len(self.values.shape)==2:
            new_log = self.__class__.two_dim(self.depth_top, self.x_axis, self.values, self.name, units=self.units, xy=(self.x, self.y))
        
        if self.elev is not None:
            new_log.elevation(self.elev)
        return new_log        

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
    
    def add_offset(self, offset):
        """add an offset to the log values"""
        for mem in vars(self).keys():
            if mem.startswith('depth'):
                exec_str = f'self.{mem} += offset'
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
            
        self.source_method = None

    def plot(self, ax=None, x_offset=0., **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        cbar = kwargs.pop('cbar', False)
        if 'cmap' in kwargs:
            cmap = kwargs.pop('cmap')
            dx = kwargs.pop('dx', 1.)
            border = kwargs.pop('border', '')
            zset = np.unique(np.concatenate((self.depth_top, self.depth_bot)))
            if self.elev is not None:
                zset = zset[::-1]

            if isinstance(dx, float) or isinstance(dx, int):
                dx = np.ones_like(zset)*dx
            cbar = True

            X = np.outer(np.array([-0.5,0.5]), dx)
            Z = np.repeat(zset, 2).reshape(-1, 2).T
            V = np.repeat(self.values, 1).reshape(self.Nz, 1).T
            pcm = ax.pcolormesh(X+x_offset, Z, V,  cmap=cmap, **kwargs)
            tmpz = Z

            if border != '':
                ax.plot(X[0]+x_offset, Z[0], border)
                ax.plot(X[1]+x_offset, Z[1], border)
                ax.plot(X[:,0]+x_offset, Z[:,0], border)
                ax.plot(X[:,-1]+x_offset, Z[:,-1], border)

        elif np.all(self.depth_bot == self.depth_top):
            ax.plot(self.values+x_offset, self.z, label=self.name, **kwargs)
            tmpz = self.z
        else:
            vmin, vmax = kwargs.pop('vmin', self.values.min()), kwargs.pop(
                'vmax', self.values.max())
            # this gives the plot a step-wise form, where the layer boundaries are correct.
            tmpz = np.vstack((self.depth_top, self.depth_bot)).T.flatten()
            tmpx = np.vstack((self.values, self.values)).T.flatten()
            # kwargs.setdefault('label', self.name)
            ax.plot(tmpx+x_offset, tmpz,  **kwargs)
            kwargs['vmin'], kwargs['vmax'] = vmin, vmax
        if self.units != '':
            ax.set_xlabel(self.name + f' [{self.units}]')
        else:
            ax.set_xlabel(self.name)

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
        depth_bot = np.append(depth, kwargs.pop('bottom', 1.1*depth.max()))
        self = cls(values, depth_top, depth_bot, name, **kwargs)
        self.source_method = 'standard'
        return self

    @classmethod
    def geology(cls, geology, depth_top, depth_bottom, name='geology', color_dictionary=None, **kwargs):
        self = cls(geology, depth_top, depth_bottom, name, **kwargs)
        self.source_method = 'geology'

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

            if label:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                axgeo = ax.secondary_yaxis('left')
                axgeo.set_yticks(self.z)
                labels = [x.replace(' ', '\n') for x in self.values]
                
                # axgeo.set_yticklabels(labels)
                axgeo.set_yticklabels(labels, rotation=45); axgeo.tick_params(axis='y', pad=-2)

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
            cbar = kwargs.pop('cbar', False)
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            pcm=ax.pcolor(self.x_axis, self.z, self.values, **kwargs)
            
            if cbar: res = (ax,pcm)
            else: res = ax
            return res

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
    
    def add_offset(self, offset):
        """add an offset to the log values"""
        for logs in self.logs:
            logs.add_offset(offset)
        return self

    def __iter__(self):
        return iter(self.logs)

    @property
    def names(self):
        return [x.name for x in self.logs]

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
            fig = axs[0].figure

        if isinstance(axs, matplotlib.axes._axes.Axes):
            axs = np.array([axs])

        for i, log in enumerate(self.logs):
            log.plot(ax=axs[i])
            # axs[i].set_title(log.name)
            if self.elev != 0:
                ax2 = axs[i].secondary_yaxis(
                    'right', functions=(self.elev2depth, self.depth2elev))
            if isinstance(log, Log) and log.source_method == 'geology':
                axs[i].set_aspect(0.25)
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
                tmp_log = Log(raw[name], raw['depth']-0.25 *
                              0.5, raw['depth']+0.25*0.5, name)
                logs.append(tmp_log)

        SE_decay = np.genfromtxt(export_folder+'_SE_decay.txt')
        SE_time = np.genfromtxt(export_folder+'_SE_decay_time.txt')
        # bit of a hack
        logs.append(Log.two_dim(logs[-1].z, SE_time*1000,
                    SE_decay[:, :-1], 'SE decay'))

        T2_dist = np.genfromtxt(export_folder+'_T2_dist.txt')*100
        T2_dist_bins = 10**np.genfromtxt(export_folder+'_T2_bins_log10s.txt')
        # bit of a hack
        logs.append(Log.two_dim(
            logs[-1].z, T2_dist_bins, T2_dist[:, 1:], 'T2 dist'))

        super().__init__(logs, **kwargs)

        tmp = self['SE decay']
        se_fwr = tmp.copy()
        se_fwr.values = self.t2dist_forward()
        se_fwr.name = 'SE synth'

        se_res = se_fwr.copy()
        se_res.values = tmp.values-se_fwr.values
        se_res.name = 'SE residual'


        misfit = np.mean((tmp.values - se_fwr.values)**2 / self['noise'].values[:,None]**2, axis=1)
        misfit = Log(misfit, self['totalf'].depth_top, self['totalf'].depth_bot, 'misfit')

        self.logs.append(se_fwr)
        self.logs.append(se_res)
        self.logs.append(misfit)

        self.n_logs = len(self.logs)


        self.export_folder = export_folder

    def plot_wc(self, ax=None, legend=True):
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
            ax.legend(fontsize='small')
        ax.set_xlim(0, 1.)
        ax.set_xlabel('Water Content [ratio]')
        ax.set_ylabel('Depth [m]')
        return ax

    def plot(self, axs=None):
        n_extra = len(self.logs)-self.n_logs
        if axs is None:

            width_ratios = [1]*n_extra + [2, 2, 2, 1, 0.75]
            fig, axs = plt.subplots(
                1, n_extra+5, sharey=True, width_ratios=width_ratios, figsize=(11.69,8.27),layout='constrained')
        else:
            assert len(axs.flatten(
            )) >= 5, "not enough subplots provided for a dart logging data display"
            fig = axs.flatten()[0].figure

        for i in range(n_extra):
            self.logs[i].plot(ax=axs[i])
        self.plot_wc(ax=axs[n_extra])
        axs[n_extra].locator_params(axis='x',nbins=8)
        #axs[n_extra].set_axisbelow(True)
        axs[n_extra].grid(visible=True,which='major',axis='both')
        axs[n_extra].grid(visible=True,which='minor',axis='x')
        
        #self['SE decay'].x_axis *= 1000
        _, pcm2 = self['SE decay'].plot(ax=axs[n_extra+1], cbar=True)
        axs[n_extra+1].set_xlabel('SE decay [ms]')
        plt.colorbar(pcm2, ax=axs[n_extra+1], orientation='horizontal',location='top',label='Amplitude [%]')
        
        _, pcm = self['T2 dist'].plot(ax=axs[n_extra+2], cmap='Blues', cbar=True);
        cb = plt.colorbar(pcm, ax=axs[n_extra+2], orientation='horizontal',location='top',label='Water Content [ratio]')
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.locator = tick_locator
        cb.update_ticks()
        axs[n_extra+2].grid(True)

        self['mlT2'].plot(ax=axs[n_extra+2], color='r')
        axs[n_extra+2].set_xlabel('T2 dist [s]')
        axs[n_extra+2].set_xscale('log')
        axs[n_extra+2].axvline(0.003,linestyle='dashed')
        axs[n_extra+2].axvline(0.033,linestyle='dashed')

        tmp_logs = self.names
        tmp_logs.pop(tmp_logs.index('T2 dist'))
        tmp_logs.pop(tmp_logs.index('SE decay'))
        tmp_logs.pop(tmp_logs.index('noise'))

        for param in ['Ksdr', 'Ktc', 'Ksoe']:
            axs[n_extra+3].semilogx(self[param].values,
                                    self[param].z, drawstyle='steps-mid', label=param)
        axs[n_extra+3].set_xlabel('K [m/day]')
        axs[n_extra+3].set_axisbelow(True)
        axs[n_extra+3].grid(True)

        axs[n_extra+3].legend(fontsize='x-small')
        #axs[n_extra+4].legend(fontsize='x-small')
        self['noise'].plot(ax=axs[n_extra+4])
        axs[n_extra+4].set_xlabel('noise [%]')
        axs[n_extra+4].set_xlim(0., 20)
        axs[n_extra+0].set_ylim(self['totalf'].z.max(),
                                self['totalf'].z.min())
        axs[n_extra+4].yaxis.set_label_position("right")
        axs[n_extra+4].yaxis.tick_right()
        axs[n_extra+4].yaxis.set_label_text('depth [m]')
        axs[n_extra+4].set_axisbelow(True)
        axs[n_extra+4].grid(True)
        return axs

    def t2_trim(self, T2_min):
        # new_bh = self.copy()
        tmp = self['T2 dist'].copy()
        five_ms = tmp.x_axis >= T2_min
        lg1 = Log.two_dim(
            tmp.z, tmp.x_axis[five_ms], tmp.values[:, five_ms], 'T2 dist')
        mlT2 = 10**(np.sum(np.log10(lg1.x_axis[None, :])
                    * lg1.values, axis=1) / np.sum(lg1.values, axis=1))
        lg2                                    = Log(mlT2, tmp.z, tmp.z, 'mlT2')
        lg3                                    = Log(np.sum(lg1.values, axis=1), tmp.z, tmp.z, 'totalf')
        self.logs[self.names.index('T2 dist')] = lg1.copy()
        self.logs[self.names.index('mlT2')]    = lg2.copy()
        self.logs[self.names.index('totalf')]  = lg3.copy()

    def t2dist_forward(self):
        times = self['SE decay'].x_axis
        T2val =  self['T2 dist'].x_axis
        K = np.exp(-times[:,None]/T2val[None,:])
        return 100.*(K @ self['T2 dist'].values.T).T
    
    def fit_monoexponential(self, smooth_data=None):
        data = self['SE decay'].values
        time = self['SE decay'].x_axis
        if smooth_data is not None:
            from scipy.ndimage import convolve1d
            smooth_data = convolve1d(data, np.ones(smooth_data)/smooth_data, axis=0)
        else:
            smooth_data = data*1.        
        x = time[1:]
        xm = time.mean()
        dx = x-xm
        y = np.log(smooth_data[:, 1:])
        ym = np.nanmean(y, axis=1, keepdims=True)
        dy = y-ym

        rT2 = np.nansum(dx[None,:]*dy, axis=1) / np.nansum(dx[None,:]**2, axis=1)
        T2 = -1/rT2
        WC = np.exp(np.squeeze(ym) - rT2*xm)/100.
        fit_data = 100*WC[:,None]*np.exp(-self['SE decay'].x_axis[None,:]/T2[:,None])

        monoWC = self['totalf'].copy()
        monoWC.values = WC
        monoWC.name = 'mono WC'

        monoT2 = self['mlT2'].copy()
        monoT2.values = T2
        monoT2.name = 'mono T2'

        se_mono = self['SE decay'].copy()
        se_mono.values = fit_data
        se_mono.name = 'SE mono synth'

        se_mono_res = self['SE decay'].copy()
        se_mono_res.values = self['SE decay'].values-se_mono.values
        se_mono_res.name = 'SE mono residual'

        mono_misfit = np.mean((self['SE decay'].values - se_mono.values)**2 / self['noise'].values[:,None]**2, axis=1)
        mono_misfit = Log(mono_misfit, self['totalf'].depth_top, self['totalf'].depth_bot, 'mono misfit')

        for new_log in [monoWC, monoT2, se_mono, se_mono_res, mono_misfit]:
            if new_log.name in self.names:
                idx = self.names.index(new_log.name)
                self.logs[idx] = new_log
            else:
                self.logs.append(new_log)
                self.n_logs += 1
        return WC,T2

class ProjectionLine:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        dx,dy = x[1]-x[0], y[1]-y[0]
        self.phi = -np.arctan2(dy,dx)
        self.R = np.array([[np.cos(self.phi), -np.sin(self.phi)],
                           [np.sin(self.phi), np.cos(self.phi)]])
        self.iR = np.array([[np.cos(-self.phi), -np.sin(-self.phi)],
                            [np.sin(-self.phi), np.cos(-self.phi)]])

        self.length = np.sqrt(dx**2 + dy**2)
        self.m = dy/dx
        self.b = y[0] - self.m*x[0]

    def inbounds(self, x, y, buffer=np.inf):
        xp,yp = self.to_line_coords(x,y)
        inbounds = (xp>=0) * (xp<=self.length)
        inbounds *= np.abs(yp) <= buffer
        return inbounds
        
    def to_line_coords(self, x, y):
        xt,yt = x-self.x[0], y-self.y[0]
        return np.dot(self.R, np.array([xt,yt]))
    
    def from_line_coords(self, xp, yp):
        xt,yt = np.dot(self.iR, np.array([xp,yp]))
        return xt+self.x[0], yt+self.y[0]
    

    
class PieceWiseLine:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_segments = len(x)-1
        self.lines = [ProjectionLine(x[i:i+2], y[i:i+2]) for i in range(self.n_segments)]
        self.cumulative_length = np.cumsum([0]+[line.length for line in self.lines])

    @classmethod
    def from_point_list(cls, points):
        x = np.array([point[0] for point in points])
        y = np.array([point[1] for point in points])
        return cls(x,y)
    
    @property
    def length(self):
        return self.cumulative_length[-1]
    
    @property
    def m(self):
        return np.array([line.m for line in self.lines]).squeeze()

    @property
    def b(self):
        return np.array([line.b for line in self.lines]).squeeze()
        
    def inbounds(self, x, y, buffer=np.inf):
        inbounds = np.zeros_like(x, dtype=bool)
        for line in self.lines:
            inbounds += line.inbounds(x,y, buffer=buffer)
        return inbounds

    def to_line_coords(self, x, y, return_segment_index=False):
        if not isinstance(x, np.ndarray): x = np.array([x], dtype=float)
        if not isinstance(y, np.ndarray): y = np.array([y], dtype=float)
        projections = np.zeros((self.n_segments, 2, len(x)))
        projection_dist = np.zeros_like(projections)
        for i,line in enumerate(self.lines):
            projections[i] = line.to_line_coords(x,y)
            projection_dist[i] = projections[i]
            # not in bound points lie beyond the ends of the line segement.
            # There are instances when the extension of a particular line segement gets closer to a point than the line segment that is actually closest to the point. To avoid this, out of bound points get additional perpendicular distance added to them, so the ideal line segment is more likely to get assigned.
            not_inbounds = np.logical_not(line.inbounds(x,y))
            
            projection_dist[i,1,not_inbounds] = np.abs(projection_dist[i,1,not_inbounds]) + np.abs(projections[i,0,not_inbounds])
        idx_min = np.nanargmin(np.abs(projection_dist[:,1:,:]), axis=0)

        projections = idx_min.choose(projections)

        projections[0,:] += np.squeeze(self.cumulative_length[idx_min])
        if return_segment_index:
            return projections[0], projections[1], idx_min[0]
        else:
            return projections[0], projections[1]
    
    
    def from_line_coords(self, xp, yp):
        if isinstance(xp, (int, float)):
            xp, yp = np.array([xp], dtype=float), np.array([yp], dtype=float)
        idx = np.clip(np.searchsorted(self.cumulative_length, xp, side='right')-1, 0, self.n_segments-1)
        x = np.zeros_like(xp)
        y = np.zeros_like(yp)
        for i in range(len(xp)):
            x[i], y[i] = self.lines[idx[i]].from_line_coords(xp[i]-self.cumulative_length[idx[i]], yp[i])
        return x,y
    
    @property
    def line_x(self):
        xp,yp = self.to_line_coords(self.x, self.y)
        return xp
    
    @property
    def line_y(self):
        xp,yp = self.to_line_coords(self.x, self.y)
        return yp


class Section:
    X = None
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
        self.profile = PieceWiseLine.from_point_list(proj_pnts)

    def __iter__(self):
        return iter(self.logs)

    def dist_func(self, x, y, return_perp=False):
        xp,yp = self.profile.to_line_coords(x, y)
        if return_perp: return xp,yp
        else: return xp

    def dist_func_inv(self, d):
        return self.profile.from_line_coords(d, d*0.)

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
        return self.dist_func(self.x, self.y)

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
            xbnds[0] = min(x_offset-dx, xbnds[0])
            xbnds[1] = max(x_offset+dx, xbnds[1])
            ax = log.plot(ax=ax, x_offset=x_offset, **kwargs)

        if offset:
            axs1[0].set_xlim(xbnds)

        if kwargs.get('cbar', True)=='return':
            return axs1, ax[1]
        elif not isinstance(ax, plt.Axes):
            plt.colorbar(ax[1], ax=ax[0], fraction=0.01)
        else:
            return axs1

    def plot_interpolated(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if self.X is None:
            self.X, self.Z, self.Val = self.interpolate_xsection(max_bin_skip=kwargs.pop('max_bin_skip', 1))
        else:
            del kwargs['max_bin_skip']
        X,Z,Val = self.X, self.Z, self.Val


        cmap = kwargs.pop('cmap', 'rainbow' if self.name == 'WC' else 'seismic')
        if 'levels' in kwargs or 'contour_lines' in kwargs:
            pcm = ax.contourf(X, Z, Val, cmap=cmap, **kwargs)
            if kwargs.get('contour_lines', False):
                pcm2 = ax.contour(pcm, levels=pcm.levels,
                                  linewidth=0.1, colors='k')
                ax.clabel(pcm2, fmt='%2.0f', colors='k', fontsize=11)
        else:
            cbar = kwargs.pop('cbar', None)
            pcm = ax.pcolormesh(X, Z, Val, shading='auto', cmap=cmap, **kwargs)
            kwargs['cbar']=cbar

        if kwargs.get('cbar', True)=='return':
            return ax, pcm
        elif kwargs.get('cbar', True):
            label = self.name
            if not self.units == '':
                label += f' [{self.units}]'
            plt.colorbar(pcm, ax=ax, fraction=0.01, cmap=cmap, label=label)
        xp,yp = self.dist_func(self.profile.x, self.profile.y, return_perp=True)
        line_length  = xp.max()-xp.min()
        ax.set_xlim(xp.min()-0.05*line_length, xp.max()+0.05*line_length)       
        return ax

    def interpolate_xsection(self, max_bin_skip=1, search_radius=None):
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
            x_dist,perp_dist = self.dist_func(self.x, self.y, return_perp=True)
            perp_dist = np.abs(perp_dist)
            xp,yp = self.dist_func(self.profile.x, self.profile.y, return_perp=True)
            line_length  = xp.max()-xp.min()
            if search_radius is None:
                search_radius = 0.05*line_length

            avg_dx = 3*line_length / sum(np.abs(perp_dist)<search_radius)
            x_mesh = np.arange(xp.min(), xp.max(), avg_dx)

            all_z = np.linspace(30, -30, 60)

            log_redisc = Log(0.*all_z[:-1], all_z[1:], all_z[:-1], 'redisc')
            log_redisc.elevation(0.)

            near_logs = []
            for i,lg in enumerate(self):
                if perp_dist[i]<search_radius:
                    clog = lg.rediscretize(log_redisc)
                    clog.x = lg.x
                    clog.y = lg.y
                    near_logs.append(clog)



            
            xsec = np.zeros(x_mesh.shape+log_redisc.z.shape)
            n = np.zeros_like(xsec)


            x = np.array([lg.x for lg in near_logs])
            y = np.array([lg.y for lg in near_logs])
            x_dist,perp_dist = self.dist_func(x, y, return_perp=True)
            perp_dist = np.abs(perp_dist)
            print('number of close logs are: ', len(near_logs))
            X,Z = np.meshgrid(x_mesh, log_redisc.z, indexing='ij')


            for ix,x in enumerate(x_mesh):
                r = np.sqrt((x_dist-x)**2 + perp_dist**2)
                in_circle = (r<search_radius)*((x_dist-x)**2 < avg_dx**2)
                weight = np.nan_to_num(1./r)
                weight = weight/np.nansum(weight[in_circle])
                
                if np.sum(in_circle.astype(int))>0:
                    for i,lg in enumerate(near_logs):
                        if in_circle[i]:
                            xsec[ix] += np.nan_to_num(lg.values*weight[i])
                            n[ix]    += (lg.values/lg.values)*weight[i]
            xsec /= n
            xsec[xsec==0.] = np.nan
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
            
            X = X[keep, :]
            Z = Z[keep, :]
            xsec = xsec[keep, :]
            X,Z = np.meshgrid(X[:,0], log_redisc.z)
            return X,Z, xsec.T
