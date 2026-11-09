"""Generate the manuscript battery variance and recovery figures."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

import figure23_panels as panels
from recovery_analysis import FIGURE3_CELL, battery_diagnostics, comparison_intervals, recovery_trace

DATASETS = ['MIT', 'HUST', 'CityU', 'ISEA', 'Tongji']
BLUE, DARK, GREY, EARLY, NEAR = '#8DAEC8', '#648DB0', '#B6BDC4', '#EDF0F3', '#F4EBE0'
COLORS = ['#527D9D', '#7AA9B3', '#B49B80', '#8792B6', '#A795A7']

STYLE = {
    'font.family': 'Arial', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': .65,
    'axes.edgecolor': '#34383C', 'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'axes.unicode_minus': False,
    'path.simplify': False, 'xtick.major.width': .65, 'ytick.major.width': .65,
    'xtick.major.size': 2.4, 'ytick.major.size': 2.4,
}


def fig(width, height):
    figure = plt.figure(figsize=(width / 25.4, height / 25.4))
    figure._mm = (width, height)
    return figure


def ax(figure, x, y, width, height):
    page_width, page_height = figure._mm
    axis = figure.add_axes([x / page_width, 1 - (y + height) / page_height,
                            width / page_width, height / page_height])
    axis.spines[['top', 'right']].set_visible(False)
    axis.tick_params(pad=2, length=2.5, width=.65)
    return axis


def text(figure, x, y, label, size=8.5, **style):
    page_width, page_height = figure._mm
    return figure.text(x / page_width, 1 - y / page_height, label, fontsize=size,
                       va=style.pop('va', 'top'), color='black', **style)


def letter(figure, label, x, y):
    text(figure, x, y, label, 13, weight='bold')


def save(figure, number, output_dir):
    figure.canvas.draw()
    for label in figure.findobj(matplotlib.text.Text):
        if label.get_text() and label.get_fontsize() < 12:
            label.set_fontsize(max(9.2, label.get_fontsize() * 1.04))
    for label in figure.texts:
        if label.get_text() in list('abcdefghi') and label.get_weight() in ['bold', 700]:
            label.set_fontsize(12)
            label.set_ha('left')
            label.set_va('top')
    for fmt in ('pdf', 'svg', 'png'):
        figure.savefig(output_dir / f'Figure_{number}.{fmt}', dpi=600, facecolor='white')
    plt.close(figure)


def box(axis, values, positions, colors, width=.45, pts=6, jitter=.19):
    for values_at_position, position, color in zip(values, positions, colors):
        v = np.asarray(values_at_position, float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        axis.boxplot([v], positions=[position], widths=width, manage_ticks=False,
                     showfliers=False, whis=(0, 100), patch_artist=True,
                     boxprops={'facecolor': matplotlib.colors.to_rgba(color, .24),
                               'edgecolor': '#7D8791', 'lw': .6},
                     medianprops={'color': 'black', 'lw': 1},
                     whiskerprops={'color': '#A7B0B9', 'lw': .5},
                     capprops={'color': '#A7B0B9', 'lw': .5})
        offset = jitter * np.sin(np.arange(len(v)) * 2.39996)
        axis.scatter(position + offset, v, s=pts, c=color, alpha=.68,
                     edgecolors='white', linewidths=.2, zorder=3)


def trace_variance(figure, cell, window, panel, x, y, export):
    clock, state, knee = np.asarray(cell['clock']), np.asarray(cell['state']), cell['tau']
    variance = pd.Series(state).rolling(window).var(ddof=1).to_numpy()
    intervals = comparison_intervals(clock, knee)
    axis = ax(figure, x, y, 44, 33)
    for period, color in [('early', EARLY), ('near', NEAR)]:
        lo, hi = intervals[period]
        axis.axvspan(lo, hi, color=color, lw=0)
    axis.plot(clock, np.where(variance > 0, variance, np.nan), color=DARK, lw=.60)
    for lo, hi in intervals.values():
        median = np.nanmedian(variance[(clock >= lo) & (clock < hi)])
        axis.hlines(median, lo, hi, color='black', lw=1)
    axis.axvline(knee, color='black', lw=.7, ls=(0, (3, 2)))
    panels.logarithmic_range(axis, variance, count=3, pad=.15)
    axis.set_xlim(clock[0], clock[-1])
    panels.sparse_clock(axis)
    axis.set_xlabel('Cycle number')
    axis.set_ylabel('$V(A)$')
    for lo, hi in panels.runs(np.isfinite(variance) & (variance == 0)):
        axis.plot([clock[lo], clock[hi]], [.025, .025], color=GREY, lw=.7,
                  transform=axis.get_xaxis_transform())
    pd.DataFrame({'clock': clock, 'A': state, 'variance': variance}).to_csv(
        export / f'Figure2_{panel}.csv', index=False)
    return axis


def _figure2(cells, variance, output_dir, export):
    # Figure 2: principle, fixed five examples, incidence (g) and magnitude (h).
    f = fig(190, 208)
    panels.calculation(f).to_csv(export / 'Figure2_a_calculation.csv', index=False)
    for t in f.texts:
        if t.get_text()=='a':t.set_position((3/190,1-3/208))
    key=ax(f,17,44.5,168,6);key.set_axis_off()
    key.legend(handles=[Patch(facecolor=EARLY,label='Earlier'),Patch(facecolor=NEAR,label='Near knee'),
                        Line2D([],[],color='black',lw=1,label='Interval median'),
                        Line2D([],[],color='black',lw=.7,ls='--',label='Knee'),
                        Line2D([],[],color=GREY,lw=.7,label='Zero variance')],
               ncol=5,frameon=False,loc='center',fontsize=8,handlelength=1.45,
               handletextpad=.5,columnspacing=1.45)
    examples = [
        ('MIT', 'b39', 'MIT', 'b', 17, 57),
        ('HUST2', '6-8', 'HUST', 'c', 79, 57),
        ('CityU', '811-Li-D1-25-3', 'CityU', 'd', 141, 57),
        ('ISEA', 'Cell33', 'ISEA', 'e', 17, 108),
        ('Tongji', 'CY25-025_1-#4', 'Tongji', 'f', 79, 108),
    ]
    windows = variance.set_index(['batch', 'id']).variance_window
    for batch, sid, dataset, panel, x, y in examples:
        cell = cells[(batch, sid)]
        axis = trace_variance(f, cell, int(windows.loc[(batch, sid)]), panel, x, y, export)
        letter(f, panel, x - 14, y - 7.5)
        text(f, x + 22, y - 6.6, dataset, 9.5, weight='bold', ha='center')
        axis.tick_params(labelsize=7.6)

    # g: paired horizontal rainclouds of raw interval-median variance.
    # Dataset is categorical on the left axis; density height has no quantitative role.
    letter(f,'g',127,100.5)
    g=ax(f,141,111,44,84);zero_x=-11.15
    g.set(xlim=(-11.4,-.7),ylim=(-.58,4.58))
    g.set_yticks([4,3,2,1,0],DATASETS,fontsize=7.5)
    g.tick_params(axis='y',length=2,pad=2)
    g.set_xticks([zero_x,-9,-6,-3,-1],['0',r'$10^{-9}$',r'$10^{-6}$',r'$10^{-3}$',r'$10^{-1}$'])
    g.tick_params(axis='x',labelsize=7.1);g.set_xlabel('Median $V(A)$',fontsize=8)
    key=ax(f,138,102,49,5);key.axis('off')
    key.legend(handles=[Patch(facecolor='#DDE5EB',edgecolor='#8798A7',lw=.55,label='Earlier'),
                        Patch(facecolor='#ECD4C6',edgecolor='#C69277',lw=.55,label='Near knee')],
               ncol=2,frameon=False,loc='center',fontsize=7.2,handlelength=.9,
               handletextpad=.35,columnspacing=.85,borderaxespad=0)
    records=[]
    for j,ds in enumerate(DATASETS):
        rr=variance[(variance.dataset==ds)&variance.variance_evaluable].copy()
        centre=4-j
        for col,y,c,edge in [('early_raw_median',centre+.23,'#DDE5EB','#8798A7'),
                             ('near_raw_median',centre-.23,'#ECD4C6','#C69277')]:
            vals=rr[col].to_numpy();logs=np.log10(vals[vals>0])
            xx=np.linspace(-10.7,-.72,230);density=gaussian_kde(logs)(xx)
            density=density/density.max()*.245
            g.fill_between(xx,y+.025,y+.025+density,facecolor=c,edgecolor='none')
            g.plot(xx,y+.025+density,color=edge,lw=.55)
            disp=np.full(len(vals),zero_x);np.log10(vals,out=disp,where=vals>0)
            # Deterministic spread avoids hiding observations at similar x values.
            bins=np.round(disp/.11).astype(int);offset=np.zeros(len(vals))
            for bn in np.unique(bins):
                ids=np.flatnonzero(bins==bn);order=np.argsort(disp[ids],kind='stable');ids=ids[order]
                offset[ids]=np.linspace(-.065,.065,len(ids)) if len(ids)>1 else 0
            g.scatter(disp,y-.10+offset,s=4.0,c=edge,alpha=.62,edgecolors='white',linewidths=.12,zorder=3)
            q=np.quantile(vals,[.25,.5,.75]);qq=np.where(q>0,np.log10(np.maximum(q,np.finfo(float).tiny)),zero_x)
            g.plot([qq[0],qq[2]],[y+.025,y+.025],color=edge,lw=2.2,solid_capstyle='butt',zorder=4)
            g.scatter(qq[1],y+.025,s=8,facecolors='white',edgecolors='#38434B',lw=.6,zorder=5)
        records.extend(rr[['dataset','id','early_raw_median','near_raw_median','raw_median_ratio']].to_dict('records'))
    # The isolated zero column is separated from the positive log scale.
    g.plot([.057,.070],[-.006,.006],transform=g.transAxes,color='#34383C',lw=.55,clip_on=False)
    g.plot([.073,.086],[-.006,.006],transform=g.transAxes,color='#34383C',lw=.55,clip_on=False)
    if len(records) != 301 or sum(row['raw_median_ratio'] > 1 for row in records) != 267:
        raise ValueError("Unexpected Figure 2 cohort counts.")
    pd.DataFrame(records).to_csv(export/'Figure2_g_paired_variance.csv',index=False)

    # h: matched within-cell amplification, preserving exact zeros in a separate row.
    letter(f,'h',3,150)
    a=ax(f,17,158,105,37);zero_y=-5.25
    ratios=[]
    for j,ds in enumerate(DATASETS):
        r=variance[(variance.dataset==ds)&variance.variance_evaluable]
        vals=r.raw_median_ratio.to_numpy();y=np.full(vals.shape,zero_y)
        np.log10(vals,out=y,where=vals>0)
        box(a,[y],[j],[BLUE],width=.45,pts=7,jitter=.22)
        ratios.extend(r[['dataset','id','raw_median_ratio']].to_dict('records'))
    a.axhline(0,color=GREY,lw=.7,ls='--',zorder=0)
    a.set(xlim=(-.55,4.55),ylim=(-5.55,4.6));a.set_xticks(range(5),DATASETS,fontsize=8)
    a.set_yticks([zero_y,-4,-2,0,2,4],['0',r'$10^{-4}$',r'$10^{-2}$','1',r'$10^2$',r'$10^4$'])
    a.set_ylabel('Near / earlier variance')
    a.spines['left'].set_bounds(zero_y,4.6)
    a.plot([-.007,.007],[-4.62,-4.49],transform=a.get_yaxis_transform(),color='#34383C',lw=.6,clip_on=False)
    a.plot([-.007,.007],[-4.48,-4.35],transform=a.get_yaxis_transform(),color='#34383C',lw=.6,clip_on=False)
    pd.DataFrame(ratios).to_csv(export/'Figure2_h_cell_ratios.csv',index=False)
    save(f, 2, output_dir)


def _figure3(cells, variance, recovery, output_dir, export):
    # Figure 3: theory, one worked example and two population diagnostics.
    f = fig(190, 190)
    cell = cells[FIGURE3_CELL]
    clock, state = np.asarray(cell['clock']), np.asarray(cell['state'])
    knee = float(cell['tau'])
    trace_data = pd.DataFrame({'clock': clock, 'state': state})
    proxy = recovery_trace(clock, state, np.flatnonzero(clock <= knee), innovation_diagnostics=True)
    trace = trace_data.merge(proxy, on='clock', how='left')
    trace.insert(0, 'batch', FIGURE3_CELL[0])
    trace.insert(1, 'id', FIGURE3_CELL[1])
    trace.insert(2, 'knee', knee)
    trace.to_csv(export / 'Figure3_c_recovery.csv', index=False)
    var = variance.set_index(['batch', 'id']).loc[FIGURE3_CELL]
    rec = recovery.set_index(['batch', 'id']).loc[FIGURE3_CELL]
    intervals = comparison_intervals(clock, knee)
    case = {'transition': knee,
            'intervals': {'earlier': intervals['early'], 'near': intervals['near']},
            'recovery_primary_statistics': [],
            'mean_variances': []}
    for period, column in [('earlier', 'early'), ('near', 'near')]:
        case['recovery_primary_statistics'].append(
            dict(period=period, kappa_median=rec[column + '_kappa']))
        case['mean_variances'].append(
            dict(period=period, **{f'{quantity}_relative': var[f'{column}_{quantity}'] / var.early_raw
                                   for quantity in ('raw', 'trend', 'residual')}))
    panels.draw_mechanism(f)
    panels.draw_analytic_readout(f)
    panels.draw_empirical(f, trace_data, proxy, case, 'c', 3)

    c_axes = [axis for axis in f.axes if getattr(axis, '_panel_owner', None) == 'c']
    for panel, top in zip(c_axes, (86, 112, 156, 156, 143)):
        bounds = panel.get_position()
        height = {112: 25/190, 156: 23/190}.get(top, bounds.height)
        panel.set_position([bounds.x0, 1-top/190-height, bounds.width, height])
    c_axes[1].set_yticks([1e-2, 1e-1, 1])
    for trace in c_axes[:2]:
        trace.set_xlim(0, knee + 42)
        trace.set_xticks([0, 250, 500, 750])
    c_axes[0].legend(handles=[Patch(facecolor=panels.EARLY,alpha=.3,label='Earlier'),
                            Patch(facecolor=panels.NEAR,alpha=.3,label='Near knee')],
                     loc='lower left',bbox_to_anchor=(.11,.04),ncol=2,
                     frameon=False,fontsize=7.1,handlelength=1.0,
                     handletextpad=.4,columnspacing=1.2,borderaxespad=0)
    good = recovery[recovery.evaluable]
    letter(f, 'd', 99, 78)
    a = ax(f, 116, 86, 67, 40)
    box(a, [good.loc[good.dataset == ds, 'kappa_ratio'] for ds in DATASETS],
        range(5), COLORS, pts=5.5)
    a.set_yscale('log')
    a.axhline(1, color=GREY, ls='--', lw=.7)
    a.set_xticks(range(5), DATASETS, fontsize=8)
    a.set_ylabel(r'Near / earlier $\kappa_{\mathrm{eff}}$')
    a.set_xlim(-.6, 4.6)
    a.set_yticks([.01,1,100])
    a.yaxis.set_minor_locator(ticker.NullLocator())

    letter(f, 'e', 99, 133)
    a = ax(f, 116, 143, 67, 35)
    xx, yy = good.kappa_ratio.to_numpy(), good.residual_ratio.to_numpy()
    xmin, xmax = min(xx) * .6, max(xx) * 1.7
    ymin, ymax = min(yy) * .6, max(yy) * 1.6
    a.set(xscale='log', yscale='log', xlim=(xmin, xmax), ylim=(ymin, ymax))
    a.fill_between([xmin, 1], 1, ymax, color=BLUE, alpha=.12, lw=0)
    a.axvline(1, color=GREY, lw=.65, ls='--')
    a.axhline(1, color=GREY, lw=.65, ls='--')
    for ds, color in zip(DATASETS, COLORS):
        group = good[good.dataset == ds]
        a.scatter(group.kappa_ratio, group.residual_ratio, s=9, c=color,
                  edgecolors='white', lw=.2, alpha=.75)
    a.set_xlabel('Near / earlier recovery-rate proxy')
    a.set_ylabel('Near / earlier residual variance')
    a.set_xticks([.01,1,100])
    a.set_yticks([1e-4,.01,1,100])
    for axis in (a.xaxis, a.yaxis):
        axis.set_minor_locator(ticker.NullLocator())
    joint = int(((xx < 1) & (yy > 1)).sum())
    a.text(.975, .06, f'{joint} / {len(good)}\nin shaded quadrant',
           transform=a.transAxes, ha='right', va='bottom', fontsize=8)
    key = ax(f, 114, 135, 70, 4)
    key.axis('off')
    key.legend(handles=[Line2D([], [], ls='', marker='o', color=color, label=ds, ms=3)
                        for color, ds in zip(COLORS, DATASETS)],
               ncol=5, loc='center', frameon=False, fontsize=7.6,
               handlelength=.55, handletextpad=.25, columnspacing=.7)
    good.to_csv(export / 'Figure3_de_cohort.csv', index=False)
    save(f, 3, output_dir)

    if len(good) != 247 or joint != 123:
        raise ValueError("Unexpected Figure 3 cohort counts.")
    return {
        'recovery_evaluable': len(good),
        'paired_direction': joint,
        'coverage50_evaluable': int(recovery.coverage50.sum()),
        'coverage50_paired': int(recovery.joint_coverage50.sum()),
        'example_batch': FIGURE3_CELL[0], 'example_id': FIGURE3_CELL[1],
        'example_kappa_ratio': float(rec.kappa_ratio),
        'example_detrended_variance_ratio': float(var.residual_ratio),
    }


def build_figures(output_dir, histories_by_batch=None):
    """Export Figures 2 and 3 with their cohort and panel data."""
    output_dir = Path(output_dir)
    export = output_dir / 'panel_data'
    export.mkdir(parents=True, exist_ok=True)
    cells, variance, recovery = battery_diagnostics(histories_by_batch)
    variance.to_csv(export / 'Figure2_variance_cohort.csv', index=False)
    recovery.to_csv(export / 'Figure3_recovery_cohort.csv', index=False)
    style = STYLE.copy()
    if not any(font.name == 'Arial' for font in font_manager.fontManager.ttflist):
        style.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
    with matplotlib.rc_context(style):
        _figure2(cells, variance, output_dir, export)
        results = _figure3(cells, variance, recovery, output_dir, export)
    results['variance_evaluable'] = int(variance.variance_evaluable.sum())
    results['variance_increased'] = int((variance.raw_median_ratio > 1).sum())
    return results
