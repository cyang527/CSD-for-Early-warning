"""Drawing helpers for the battery variance and recovery figures."""

import math

import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Ellipse, FancyArrowPatch, Patch, Rectangle

EARLY = '#93A6B6'
NEAR = '#D9987F'
LIGHT = '#D5D8DC'
GREY = '#A4AAB1'
FAST = '#969CA4'
SLOW = '#BB9CA0'
INK = '#000000'


def expected_window_variance(window, recovery_time):
    """Expected sample variance divided by forcing variance and sample spacing."""
    lag = np.arange(1, window)[:, None]
    recovery_time = np.asarray(recovery_time)
    return (recovery_time * np.sum((window - lag) * (-np.expm1(-lag / recovery_time)), axis=0)
            / (window * (window - 1)))


def ax_mm(fig, x, y, w, h, owner):
    PAGE_W, PAGE_H = fig._mm
    ax = fig.add_axes([x / PAGE_W, 1 - (y + h) / PAGE_H,
                       w / PAGE_W, h / PAGE_H])
    ax._panel_owner = owner
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.tick_params(pad=2.5)
    return ax


def text_mm(fig, x, y, text, owner, size=8.8, **kw):
    PAGE_W, PAGE_H = fig._mm
    txt = fig.text(x / PAGE_W, 1 - y / PAGE_H, text,
                   color=INK, fontsize=size, va='top', **kw)
    txt._panel_owner = owner
    return txt


def arrow_mm(fig, start, end, owner, color=GREY, lw=.85, style='-|>', scale=8):
    PAGE_W, PAGE_H = fig._mm
    p = FancyArrowPatch((start[0] / PAGE_W, 1 - start[1] / PAGE_H),
                        (end[0] / PAGE_W, 1 - end[1] / PAGE_H),
                        transform=fig.transFigure, arrowstyle=style,
                        mutation_scale=scale, linewidth=lw, color=color,
                        shrinkA=0, shrinkB=0)
    p._panel_owner = owner
    fig.add_artist(p)
    return p


def letter(fig, name, x, y):
    obj = text_mm(fig, x, y, name, name, size=12.5, weight='bold')
    obj.set_gid('panel-letter-' + name)


def sparse_clock(ax):
    lo, hi = ax.get_xlim()
    loc = ticker.MaxNLocator(nbins=3, min_n_ticks=2, steps=[1, 2, 2.5, 5, 10])
    ticks = loc.tick_values(lo, hi)
    ax.set_xticks(ticks[(ticks >= lo) & (ticks <= hi)])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))


def logarithmic_range(ax, values, count=3, pad=.10):
    values = np.asarray(values, float)
    v = values[np.isfinite(values) & (values > 0)]
    low, high = np.log10(v.min()), np.log10(v.max())
    span = max(.6, high-low)
    ax.set_yscale('log')
    ax.set_ylim(10 ** (low-pad*span), 10 ** (high+pad*span))
    decades = np.arange(math.ceil(low), math.floor(high)+1)
    if len(decades) > count:
        decades = decades[np.unique(np.round(np.linspace(0, len(decades)-1, count)).astype(int))]
    if len(decades):
        ax.set_yticks(10. ** decades)
    ax.yaxis.set_minor_locator(ticker.NullLocator())


def calculation(fig):
    """Two equal trailing windows map to two points at their right endpoints."""
    letter(fig, 'a', 4, 3)
    n, window = 112, 16
    t = np.arange(n)
    rng = np.random.default_rng(803)
    envelope = .03 + .09 / (1 + np.exp(-(t-69)/8))
    state = .30 + .0010*t + envelope*(.7*rng.standard_normal(n)+.7*np.sin(1.7*t))
    var = pd.Series(state).rolling(window).var(ddof=1).to_numpy()
    blue = '#7D9FBE'
    grey = '#9CA1A7'
    upper = ax_mm(fig, 17, 11, 97, 11, 'a')
    lower = ax_mm(fig, 17, 29, 97, 11, 'a')
    upper.plot(t, state, color=blue, lw=.85)
    lower.plot(t, var, color=blue, lw=.95)
    upper.set_ylim(.12, .77)
    lower.set_ylim(0, max(var[np.isfinite(var)])*1.15)
    for ax in (upper, lower):
        ax.tick_params(pad=2)
        ax.set_xlim(0, n-1)
        ax.set_xticks([])
        ax.set_yticks([])
    upper.set_ylabel('$z$', labelpad=3)
    lower.set_ylabel('$V_i$', labelpad=3)
    lower.set_xlabel('Observation index', labelpad=2)
    for right in (36, 94):
        left = right-window+1
        upper.axvspan(left, right, color='#EDF0F3', zorder=0, lw=0)
        upper.annotate('', (left, .67), (right, .67),
                       arrowprops=dict(arrowstyle='|-|', lw=.7, color=grey))
        upper.text((left+right)/2, .72, '$W$', ha='center', va='center', fontsize=10)
        lower.plot(right, var[right], 'o', ms=3.7, mfc=blue, mec='white', mew=.5, zorder=6)
        link = ConnectionPatch((right, .13), (right, var[right]),
                               'data', 'data', axesA=upper, axesB=lower,
                               color=grey, lw=.7, arrowstyle='->', shrinkB=3, zorder=5)
        link._panel_owner = 'a'
        fig.add_artist(link)
    text_mm(fig, 150, 9, 'Within each trailing window', 'a', size=10.5, ha='center')
    text_mm(fig, 150, 19, r'$V_i=\frac{1}{W-1}\sum (z_k-\bar z_i)^2$',
            'a', size=12, ha='center')
    text_mm(fig, 150, 32, 'Past observations only', 'a', size=10.5, ha='center')
    return pd.DataFrame({'index': t, 'schematic_state': state, 'schematic_variance': var})


def runs(mask):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)-1)


def draw_mechanism(fig):
    """Effective wells and equal-duration relaxation at fixed forcing."""
    letter(fig, 'a', 3, 2)
    PAGE_W, PAGE_H = fig._mm
    sigma, xi0, elapsed = 1., 1., 2.
    xx=np.linspace(-2,2,241)
    disturbance_glyph=np.array([0,.25,-.7,.6,-.15,.95,-.45,.2,-.9,
                                .5,.1,-.35,.7,-.2,.9,-.6,0])
    disturbance=ax_mm(fig,12.5,16.4,23,4.2,'a')
    disturbance.axis('off')
    disturbance.set_xlim(0,1);disturbance.set_ylim(-1.2,1.2)
    disturbance.plot(np.linspace(0,1,len(disturbance_glyph)),
                     disturbance_glyph,color=GREY,lw=.8)
    text_mm(fig,24,12.2,'Fixed noise','a',size=8.5,ha='center')
    arrow_mm(fig,(36.5,18.5),(39.5,18.5),'a',lw=.75,scale=6.5)
    system=Rectangle((40/PAGE_W,1-23/PAGE_H),45/PAGE_W,12/PAGE_H,
                     transform=fig.transFigure,facecolor='#F5F6F7',
                     edgecolor=GREY,lw=.6)
    system._panel_owner='a';fig.add_artist(system)
    text_mm(fig,62.5,12.7,'Same degrading system','a',size=9.0,
            weight='bold',ha='center')
    text_mm(fig,62.5,18.0,r'Monitored state $z$','a',size=8.0,ha='center')
    fork=ax_mm(fig,32.5,23,60,3.6,'a')
    fork.set_xlim(0,60);fork.set_ylim(3.6,0);fork.axis('off')
    fork.plot([30,30],[0,1.8],color=GREY,lw=.65)
    fork.plot([0,60],[1.8,1.8],color=GREY,lw=.65)
    for side in [0,60]:
        fork.plot([side,side],[1.8,3.6],color=GREY,lw=.65,clip_on=False)
    for x,kappa,color,title,variance_label in [
        (9,1.,EARLY,'Faster recovery phase','Smaller variance'),
        (69,.25,NEAR,'Slower recovery phase','Larger variance')]:
        cx=x+23.5
        text_mm(fig,cx,27.4,title,'a',size=9.1,weight='bold',ha='center')
        well=ax_mm(fig,x,32,47,29,'a')
        well.set_xlim(-2,2);well.set_ylim(-.42,2.22);well.axis('off')
        potential=kappa*xx**2/2
        for depth,alpha in [(.32,.06),(.24,.08),(.16,.11),(.08,.14)]:
            well.fill_between(xx,potential-depth,potential,
                              color=GREY,alpha=alpha,lw=0,zorder=1)
        well.plot(xx,potential,color=GREY,lw=1.15,zorder=3)
        well.plot([0,0],[-.38,.35],color=GREY,lw=.55,ls=(0,(2,2)),zorder=2)
        after=xi0*np.exp(-kappa*elapsed)
        path=np.linspace(xi0,after,80)
        marker_lift=.095
        well.plot(path,kappa*path**2/2+marker_lift,
                  color=color,lw=1.25,zorder=4)
        tail=after+min(.30,.65*(xi0-after))
        well.add_patch(FancyArrowPatch((tail,kappa*tail**2/2+marker_lift),
                     (after,kappa*after**2/2+marker_lift),
                     arrowstyle='-|>',mutation_scale=7.2,color=color,
                     lw=1.05,shrinkA=0,shrinkB=3.65,zorder=5))
        well.plot(xi0,kappa*xi0**2/2+marker_lift,marker='o',ms=6.1,
                  mfc='white',mec=GREY,mew=.8,zorder=6)
        well.plot(after,kappa*after**2/2+marker_lift,marker='o',ms=6.3,
                  mfc=color,mec='white',mew=.45,zorder=7)
        well.plot(after-.038,kappa*after**2/2+marker_lift+.03,marker='o',
                  ms=1.5,mfc='white',mec='none',alpha=.6,zorder=8)
        text_mm(fig,cx,60.7,'Local reference','a',size=7.5,ha='center')
        sd=sigma/np.sqrt(2*kappa)
        spread=ax_mm(fig,x,64,47,5,'a')
        spread.set_xlim(-2,2);spread.set_ylim(-.5,.5);spread.axis('off')
        spread.plot([-2,2],[0,0],color=LIGHT,lw=.55)
        spread.add_patch(Ellipse((0,0),2*sd,.65,facecolor=color,
                                 alpha=.25,edgecolor='none'))
        spread.plot([0,0],[-.4,.4],color=GREY,lw=.55,ls=(0,(2,2)))
        text_mm(fig,cx,70,variance_label,'a',size=9.1,ha='center')
    text_mm(fig,4.2,34,'Effective potential','a',size=8.0,
            ha='center',rotation=90)
    key=ax_mm(fig,21,5,91,4,'a');key.axis('off')
    key.legend(handles=[Line2D([],[],ls='none',marker='o',ms=4.8,
                             mfc='white',mec=GREY,mew=.8,label='Same initial deviation'),
                        Line2D([],[],ls='none',marker='o',ms=4.8,
                             mfc=GREY,mec='white',mew=.45,label='After equal interval')],
               loc='center',ncol=2,frameon=False,fontsize=7.8,
               handlelength=.8,handletextpad=.4,columnspacing=1.3)


def draw_analytic_readout(fig):
    """Exact finite-window expectation on a discrete evaluated grid."""
    letter(fig, 'b', 124, 2)
    r = np.logspace(-1, 3, 60)
    w = np.arange(5, 61, 5)
    values = np.array([expected_window_variance(window, r) for window in w])
    log = np.log10(r)
    redge = 10**np.r_[log[0] - np.diff(log)[0]/2,
                      (log[:-1] + log[1:])/2, log[-1] + np.diff(log)[-1]/2]
    wedge = np.r_[w[0]-2.5, (w[:-1]+w[1:])/2, w[-1]+2.5]
    cmap = LinearSegmentedColormap.from_list('muted_variance',
              ['#F5F5F4', '#DDE3E8', '#C4CDD5', '#E5C5B7', '#D9987F'])
    ax = ax_mm(fig, 139, 9, 39.5, 28, 'b')
    mesh = ax.pcolormesh(redge, wedge, values, cmap=cmap, vmin=0, vmax=10.2,
                         shading='flat', edgecolors='face', linewidth=.08,
                         antialiased=False, rasterized=False)
    ax.set_xscale('log')
    ax.set_xlim(redge[0], redge[-1])
    ax.set_ylim(2.5, 62.5)
    ax.set_xticks([.1, 10, 1000])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yticks([10, 30, 60])
    ax.set_ylabel('Window length $W$', labelpad=3)
    for window in [10, 30]:
        ax.axhline(window, color=INK, lw=.5, ls=(0, (2, 2)))
    ax.set_xlabel('Recovery time / sampling interval', labelpad=3, fontsize=8.1)
    cax = ax_mm(fig, 181.5, 9, 1.4, 28, 'b')
    cb = fig.colorbar(mesh, cax=cax)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.solids.set_linewidth(.08)
    cb.set_ticks([0, 5, 10])
    cb.ax.tick_params(labelsize=7.5, length=1.8, pad=1.8)
    cb.outline.set_linewidth(.45)
    text_mm(fig, 166, 2.5, r'$\mathrm{E}[V_W(\xi)]/(\sigma^2\Delta)$', 'b',
            size=9, ha='center')

    cut = ax_mm(fig, 139, 53, 39.5, 18, 'b')
    rr = np.logspace(-1, 3, 450)
    for window, color, style in [(10, FAST, (0, (3, 2))), (30, SLOW, '-')]:
        y = expected_window_variance(window, rr)
        cut.plot(rr, y, color=color, lw=1.3, ls=style)
        cut.text(900, y[-1] + .25,
                 '$W=' + str(window) + '$', fontsize=8.0, ha='right')
    cut.set_xscale('log')
    cut.set_xlim(.1, 1000)
    cut.set_ylim(0, 6.2)
    cut.set_xticks([.1, 10, 1000])
    cut.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
    cut.xaxis.set_minor_locator(ticker.NullLocator())
    cut.set_yticks([0, 2, 4])
    cut.tick_params(axis='x',pad=4.5)
    cut.set_xlabel('Recovery time / sampling interval', labelpad=3, fontsize=8.1)
    cut.set_ylabel(r'$\mathrm{E}[V_W]/(\sigma^2\Delta)$', labelpad=3, fontsize=8.1)


def tint(ax, meta):
    for period, color in [('earlier', EARLY), ('near', NEAR)]:
        lo, hi = meta['intervals'][period]
        ax.axvspan(lo, hi, color=color, alpha=.14, lw=0, zorder=0)


def draw_empirical(fig, data, recovery_data, meta, name, x0):
    """Matched state and recovery traces with a decomposition of sample variance."""
    letter(fig, name, x0, 78)
    full = ax_mm(fig, x0 + 15, 98, 72, 17, name)
    full.plot(data.clock, data.state, color=GREY, lw=.7, zorder=2)
    tint(full, meta)
    for period, color in [('earlier', EARLY), ('near', NEAR)]:
        lo, hi = meta['intervals'][period]
        shown = data.state.where((data.clock >= lo) & (data.clock < hi))
        full.plot(data.clock, shown, color=color, lw=.95, zorder=3)
    full.axvline(meta['transition'], color=INK, lw=.75, ls=(0, (3, 2)), zorder=4)
    full.text(meta['transition'], 1.04, r'$\tau$', transform=full.get_xaxis_transform(),
              fontsize=9, ha='center', va='bottom')
    full.set_xlim(data.clock.min(), data.clock.max())
    full.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    full.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=True))
    clock_limits = full.get_xlim()
    clock_ticks = [t for t in full.get_xticks() if clock_limits[0] <= t <= clock_limits[1]]
    full.set_xticks(clock_ticks)
    full.set_xlim(clock_limits)
    full.set_ylabel('$A$', labelpad=3)
    full.set_xlabel('Cycle number', labelpad=3)
    full.xaxis.label.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'pad': .1})
    rec = ax_mm(fig, x0 + 15, 133, 72, 20, name)
    rec.plot(recovery_data.clock, recovery_data.kappa, color=GREY, lw=.65, zorder=2)
    tint(rec, meta)
    statistics = {r['period']: r for r in meta['recovery_primary_statistics']}
    for period, color in [('earlier', '#6E8BA1'), ('near', '#B97763')]:
        lo, hi = meta['intervals'][period]
        shown = recovery_data.kappa.where((recovery_data.clock >= lo) &
                                         (recovery_data.clock < hi))
        rec.plot(recovery_data.clock, shown, color=color, lw=.85, zorder=3)
        med = statistics[period]['kappa_median']
        rec.plot([lo, hi], [med, med], color=INK, lw=1.2, zorder=4)
    rec.axvline(meta['transition'], color=INK, lw=.75, ls=(0, (3, 2)), zorder=5)
    rec.set_xscale('linear')
    rec.set_xticks(clock_ticks)
    rec.set_xlim(clock_limits)
    rec.set_yscale('log')
    rec.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
    rec.yaxis.set_minor_locator(ticker.NullLocator())
    rec.set_ylabel(r'$\kappa_{\rm eff}$ (cycle$^{-1}$)', labelpad=3, fontsize=8)

    rows = {r['period']: r for r in meta['mean_variances']}
    raw = ax_mm(fig, x0 + 19, 168, 27, 19, name)
    residual = ax_mm(fig, x0 + 65, 168, 24, 19, name)
    colors = [EARLY, NEAR]
    max_raw = max(r['raw_relative'] for r in rows.values())
    max_res = max(r['residual_relative'] for r in rows.values())
    for i, (period, color) in enumerate(zip(['earlier', 'near'], colors)):
        row = rows[period]
        raw.bar(i, row['residual_relative'], width=.46, color=color, edgecolor=INK, lw=.35)
        raw.bar(i, row['trend_relative'], bottom=row['residual_relative'],
                width=.46, color=color, alpha=.25, edgecolor=INK, lw=.35)
        raw.text(i, row['raw_relative'] + max_raw * .065,
                 f"{row['raw_relative']:.1f}" if max_raw > 10 else f"{row['raw_relative']:.2f}",
                 ha='center', va='bottom', fontsize=8.0)
        residual.bar(i, row['residual_relative'], width=.46, color=color,
                     edgecolor=INK, lw=.35)
        residual.text(i, row['residual_relative'] + max_res * .065,
                      f"{row['residual_relative']:.3f}" if max_res < 1 else f"{row['residual_relative']:.2f}",
                      ha='center', va='bottom', fontsize=8.0)
    for ax, maximum in [(raw, max_raw), (residual, max_res)]:
        ax.set_xlim(-.65, 1.65)
        ax.set_ylim(0, maximum * 1.34)
        ax.set_xticks([0, 1], ['Earlier', 'Near'])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    raw.set_ylabel('Variance /\nearlier raw', labelpad=2, fontsize=8.0)
    residual.set_ylabel('Variance /\nearlier raw', labelpad=2, fontsize=8.0)
    text_mm(fig, x0 + 32.5, 149, 'Raw variance', name, ha='center', size=8.8)
    text_mm(fig, x0 + 77, 149, 'Detrended variance', name, ha='center', size=8.8)
    key = ax_mm(fig, x0 + 17, 194, 71, 3.5, name)
    key.axis('off')
    handles = [Line2D([], [], color=INK, lw=1.2, label='Interval median'),
               Patch(facecolor=EARLY, alpha=.25, edgecolor=INK, lw=.35, label='Trend contribution')]
    key.legend(handles=handles, loc='center', ncol=2, frameon=False,
               fontsize=7.5, handlelength=1.2, columnspacing=.85, handletextpad=.4)
