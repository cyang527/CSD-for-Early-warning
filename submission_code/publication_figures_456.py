"""Build the warning, ablation and mechanical-fatigue figures from the supplied observations."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, ticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as PlotPath
from scipy.stats import gaussian_kde, hypergeom

from manuscript_replay import FATIGUE
from publication_warning_data import (
    DATASETS, GROUPS, METHODS, RULES, comparison_matrix, condition_outcomes,
    evaluation_timing, fatigue_diagnostics, matched_variance, rates, warning_case,
)
from release_replay import collect
from warning_panels import WarningPanels

BLUE = '#8DAEC8'
CORAL, GREY = '#DDA28D', '#B6BDC4'
EARLY, NEAR, ROSE = '#EDF0F3', '#F4EBE0', '#AF8EA1'
COLORS = ['#527D9D', '#7AA9B3', '#B49B80', '#8792B6', '#A795A7']
OUTCOME_COLORS = ['#D79E87', '#94B2C9', '#D8CCE3', '#D6C398', '#ADB5BE', '#F0F0F0']
STYLE = {
    'font.family': 'Arial', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.linewidth': .65,
    'text.color': '#000000', 'axes.labelcolor': '#000000', 'axes.edgecolor': '#000000',
    'xtick.color': '#000000', 'ytick.color': '#000000', 'xtick.major.size': 2.5,
    'xtick.major.width': .65, 'ytick.major.width': .65,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'axes.unicode_minus': False, 'savefig.facecolor': 'white',
    'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'lines.dash_capstyle': 'butt', 'lines.antialiased': True, 'path.simplify': False,
}


def _style():
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
    except ValueError:
        return {**STYLE, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'}
    return STYLE


def fig(width, height):
    figure = plt.figure(figsize=(width / 25.4, height / 25.4))
    figure._mm = (width, height)
    return figure


def ax(figure, x, y, width, height):
    page_width, page_height = figure._mm
    axes = figure.add_axes([x / page_width, 1 - (y + height) / page_height,
                            width / page_width, height / page_height])
    axes.spines[['top', 'right']].set_visible(False)
    axes.tick_params(pad=2, length=2.5, width=.65)
    return axes


def text(figure, x, y, value, size=8.5, **kwargs):
    width, height = figure._mm
    return figure.text(x / width, 1 - y / height, value, fontsize=size,
                       va=kwargs.pop('va', 'top'), color='black', **kwargs)


def letter(figure, value, x, y):
    return text(figure, x, y, value, 12, weight='bold')


def _export(figure, number, output_dir, enlarge=False):
    figure.canvas.draw()
    if enlarge:
        for label in figure.findobj(matplotlib.text.Text):
            if label.get_text() and label.get_fontsize() < 12:
                label.set_fontsize(max(9.1, label.get_fontsize() * 1.10))
    for label in figure.texts:
        if len(label.get_text()) == 1 and label.get_weight() == 'bold':
            label.set_fontsize(12)
            label.set_ha('left')
            label.set_va('top')
    for suffix in ('pdf', 'svg', 'png'):
        figure.savefig(output_dir / f'Figure_{number}.{suffix}', facecolor='white',
                       dpi=600 if suffix == 'png' else 'figure')
    plt.close(figure)


def box(axes, groups, positions, colors, width=.45, pts=6, jitter=.19):
    for group, pos, color in zip(groups, positions, colors):
        values = np.asarray(group, float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        axes.boxplot([values], positions=[pos], widths=width, manage_ticks=False,
                     showfliers=False, whis=(0, 100), patch_artist=True,
                     boxprops={'facecolor': matplotlib.colors.to_rgba(color, .24),
                               'edgecolor': '#7D8791', 'lw': .6},
                     medianprops={'color': 'black', 'lw': 1},
                     whiskerprops={'color': '#A7B0B9', 'lw': .5},
                     capprops={'color': '#A7B0B9', 'lw': .5})
        if pts:
            axes.scatter(pos + jitter * np.sin(np.arange(len(values)) * 2.39996),
                         values, s=pts, color=color, alpha=.68, edgecolors='white',
                         linewidths=.2, zorder=3)


def ratio_group(axes, group, pos, color):
    values = np.asarray(group, float)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    jitter = .245 * np.sin(np.arange(len(positive)) * 2.3999632297)
    axes.scatter(pos + jitter, positive, s=10, facecolor=color,
                 edgecolor='white', linewidth=.2, alpha=.73, zorder=3)
    q25, q50, q75 = np.quantile(finite, [.25, .5, .75])
    if q25 > 0:
        axes.add_patch(Rectangle((pos - .23, q25), .46, q75 - q25,
                                 facecolor=matplotlib.colors.to_rgba(color, .19),
                                 edgecolor='#62676D', lw=.6, zorder=4))
    axes.plot([pos - .23, pos + .23], [q50] * 2, color='black', lw=.95, zorder=5)


def _draw_figure4(timing, condition, case, output_dir, export):
    f = fig(190, 119)
    p4 = WarningPanels(119)
    snapshot = p4.integrated_row(f, case, 'a', 2, False)
    for title in f.texts:
        if title.get_text() != 'a':
            title.set_fontsize(10 if title.get_text() != 'Two 60-point windows' else 9.0)
    for axis in f.axes:
        axis.tick_params(labelsize=9.2)
        for label in (axis.xaxis.label, axis.yaxis.label):
            label.set_fontsize(9.7)
        for label in axis.texts:
            label.set_fontsize(9.2)
        if axis.get_legend():
            for label in axis.get_legend().texts:
                label.set_fontsize(9.2)
    bat = timing[timing.dataset != 'Soft_Magnetic_Iron'].copy()

    letter(f, 'b', 3, 60)
    outcome_colors = {
        'In stage by knee': CORAL,
        'Other by knee': BLUE,
        'Late confirmation': '#B4BDC7',
        'No alarm': '#78838F',
    }
    flow_groups = bat.assign(outcome=np.select(
        [bat.science_valid, bat.warning, bat.missed],
        ['In stage by knee', 'Other by knee', 'No alarm'],
        default='Late confirmation'))
    counts = (pd.crosstab(flow_groups.dataset, flow_groups.outcome)
              .reindex(index=DATASETS, columns=outcome_colors, fill_value=0))
    counts.to_csv(export / 'Figure4_b_outcome_counts.csv')
    a = ax(f, 3, 66, 64, 40)
    a.set(xlim=(0, 67), ylim=(31, 0))
    a.set_axis_off()
    unit = .120
    source_x, target_x, node_width = 11.4, 45.8, 1.15
    source_cursor, target_cursor = 1.0, 1.0
    offsets = {}
    labels = ['In stage', 'Other timely', 'Late', 'No alarm']
    flow_cmap=LinearSegmentedColormap.from_list('flow_count',['#DBE7EC','#8FB1C4','#9C96B1','#CE9386'])
    flow_norm=matplotlib.colors.Normalize(vmin=0,vmax=65)
    cb_axis=ax(f,16,110,30,1.5)
    cb=f.colorbar(matplotlib.cm.ScalarMappable(norm=flow_norm,cmap=flow_cmap),cax=cb_axis,
                  orientation='horizontal',ticks=[0,30,60])
    cb.solids.set_rasterized(False)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.8,length=1.8,pad=1)
    text(f,48,110.5,'Cells per flow',8.8,ha='left',va='center')
    for outcome, label, gap in zip(outcome_colors, labels, [1.55, 1.55, 2.65, 0]):
        count = counts[outcome].sum()
        height = count * unit
        offsets[outcome] = target_cursor
        a.text(target_x + node_width + 1.15, target_cursor + height / 2,
               label, va='center', fontsize=9.0)
        target_cursor += height + gap

    for dataset, row in counts.iterrows():
        a.text(source_x - 1.0, source_cursor + unit * row.sum() / 2,
               dataset, ha='right', va='center', fontsize=9.0)
        for outcome, color in outcome_colors.items():
            height = unit * row[outcome]
            if not height:
                continue
            end_y = offsets[outcome]
            x0, x1 = source_x + node_width, target_x
            mid = (x0 + x1) / 2
            points = [(x0, source_cursor), (mid, source_cursor), (mid, end_y), (x1, end_y),
                      (x1, end_y + height), (mid, end_y + height), (mid, source_cursor + height),
                      (x0, source_cursor + height), (x0, source_cursor)]
            codes = [PlotPath.MOVETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4,
                     PlotPath.LINETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4,
                     PlotPath.CLOSEPOLY]
            flow_color=flow_cmap(flow_norm(row[outcome]))
            a.add_patch(PathPatch(PlotPath(points, codes), facecolor=flow_color,
                                  edgecolor='white', lw=.22))
            a.add_patch(Rectangle((source_x, source_cursor), node_width, height, facecolor=flow_color, lw=0))
            a.add_patch(Rectangle((target_x, end_y), node_width, height, facecolor=flow_color, lw=0))
            source_cursor += height
            offsets[outcome] += height
        source_cursor += 1.18

    # c: aligned rows, per-cell lead distributions, medians and interquartile ranges.
    letter(f, 'c', 70, 60)
    a = ax(f, 82, 66, 43, 40)
    a.set(xlim=(-3, 100), ylim=(4.55, -.65))
    lead_rows=[]
    for j,(ds,color) in enumerate(zip(DATASETS,COLORS)):
        v=100*bat.loc[(bat.dataset==ds)&bat.warning,'normalized_lead'].to_numpy()
        xx=np.linspace(max(0,v.min()),v.max(),180)
        density = gaussian_kde(v)(xx)
        density = density / density.max() * .24
        a.fill_between(xx,j-.055-density,j-.055,facecolor=color,alpha=.20,lw=0)
        a.plot(xx,j-.055-density,color=color,lw=.6)
        bins = np.round(v / 2).astype(int)
        offset = np.zeros(len(v))
        for bn in np.unique(bins):
            ids=np.flatnonzero(bins==bn)
            offset[ids]=np.linspace(-.075,.075,len(ids)) if len(ids)>1 else 0
        a.scatter(v,j+.17+offset,s=4.1,c=color,alpha=.58,edgecolors='white',linewidths=.18,zorder=3)
        q=np.quantile(v,[.25,.5,.75])
        a.plot(q[[0,2]],[j-.055,j-.055],color=color,lw=2.3,solid_capstyle='butt',zorder=4)
        a.scatter(q[1],j-.055,s=12,facecolor='white',edgecolor='#303A42',lw=.7,zorder=5)
        lead_rows.append({'dataset':ds,'n':len(v),'q25_percent':q[0],'median_percent':q[1],'q75_percent':q[2]})
    a.set_yticks(range(5),DATASETS,fontsize=9.2)
    a.set_xticks([0, 50, 100])
    a.tick_params(labelsize=9.0)
    a.set_xlabel('Lead before knee (%)',fontsize=9.6)
    pd.DataFrame(lead_rows).to_csv(export/'Figure4_c_lead_summary.csv',index=False)

    # Use the same fixed cohort for the three rules.
    rules = RULES
    rule_colors=['#249DB1','#8B77AE','#CD526B']
    markers=['o','s','D']
    paired = condition.pivot(index=['batch', 'id'], columns='method', values='science_valid').reindex(columns=rules).sort_index()
    assert paired.shape==(190,3) and not paired.isna().any().any()
    assert paired.sum().tolist()==[95,62,151]
    # Uniform subsets without replacement: exact mean and pointwise central 90%
    # composition interval. This is equivalent to averaging all inclusion orders.
    N = len(paired)
    n = np.arange(N + 1)
    letter(f,'d',132,60)
    a=ax(f,148,66,38,40)
    rows=[]
    for rule,color,marker in zip(rules,rule_colors,markers):
        K = int(paired[rule].sum())
        mean = n * K / N
        lo, hi = hypergeom.ppf(.05, N, K, n), hypergeom.ppf(.95, N, K, n)
        a.fill_between(n,lo,hi,color=color,alpha=.075,lw=0)
        a.plot(n,mean,color=color,ls=(0,(1.1,2)),lw=1.0,marker=marker,
               markevery=[0,30,60,90,120,150,190],ms=3.5,mec='white',mew=.25,label=rule)
        for ni,mu,l,u in zip(n,mean,lo,hi):
            rows.append({'rule':rule,'observed_cells':int(ni),'expected_cells_warned_in_stage':mu,
                         'central_90pct_lower':int(l),'central_90pct_upper':int(u),
                         'total_evaluation_cells':N,'total_cells_warned_in_stage':K})
        assert mean[-1]==K
    pd.DataFrame(rows).to_csv(export/'Figure4_d_observed_cell_counts.csv',index=False)
    paired.reset_index().to_csv(export/'Figure4_d_paired_stage_outcomes.csv',index=False)
    a.set(xlim=(0,196),ylim=(0,165),xticks=[0,50,100,150,190],yticks=[0,40,80,120,160])
    a.set_xlabel('Observed cells',fontsize=9.6,labelpad=3)
    a.set_ylabel('Cells warned in stage',fontsize=9.6,labelpad=3)
    a.tick_params(labelsize=9.0)
    a.legend(loc='upper left',frameon=False,fontsize=9.0,handlelength=1.25,
             labelspacing=.45,handletextpad=.45,borderaxespad=.3)

    bat.to_csv(export/'Figure4_bc.csv',index=False)
    _export(f, 4, output_dir)
    return snapshot


def _draw_figure5(ab, outcomes, paired, comparison_matrix, matrix_datasets, comparison, output_dir, export):
    f = fig(190, 162)
    letter(f,'a',3,3)
    ablation_index=ab.set_index(['dataset','group'])
    for metric,title,color,top in [('warning_rate','By-knee warning rate (%)',BLUE,22),
                                   ('science_effectiveness','Stage effectiveness (%)',CORAL,61)]:
        text(f,17,top-11,title,8.0)
        values=np.array([[100*float(ablation_index.loc[(ds,group),metric])
                          for group in GROUPS] for ds in DATASETS])
        cmap = LinearSegmentedColormap.from_list(metric, ['#FAFBFC', color])
        cmap.set_bad('#F0F0F0')
        aa=ax(f,17,top,36,20)
        mesh=aa.pcolormesh(np.arange(5)-.5,np.arange(6)-.5,np.ma.masked_invalid(values),
                          cmap=cmap,vmin=0,vmax=100,shading='flat',edgecolors='white',linewidth=.7)
        aa.set(xlim=(-.5,3.5),ylim=(4.5,-.5))
        aa.set_xticks(range(4), GROUPS, fontsize=7.6)
        aa.set_yticks(range(5), DATASETS, fontsize=7.8)
        aa.xaxis.tick_top()
        aa.tick_params(length=0, pad=2)
        aa.spines[:].set_visible(False)
        for row,col in zip(*np.where(~np.isfinite(values))):
            aa.text(col,row,'NA',ha='center',va='center',fontsize=7.5)
        cb=f.colorbar(mesh,cax=ax(f,55,top,1.7,20),ticks=[0,50,100])
        cb.outline.set_linewidth(.5)
        cb.solids.set_rasterized(False)
        cb.ax.tick_params(length=1.5,pad=1,labelsize=6.7)

    letter(f, 'b', 66, 3)
    lead_axis = ax(f, 80, 22, 42, 59)
    lead_rows = []
    for j, group in enumerate(GROUPS):
        selected = outcomes[
            (outcomes.dataset != 'Soft_Magnetic_Iron')
            & (outcomes.method == group)
            & outcomes.warning
        ].copy()
        leads = selected.normalized_lead.to_numpy()
        box(lead_axis, [leads], [j], ['#D1D8DE'], pts=0)
        lead_axis.scatter(
            j + .17 * np.sin(np.arange(len(selected)) * 2.39996), leads, s=4.2,
            c=[CORAL if aligned else BLUE for aligned in selected.science_valid],
            edgecolors='white', lw=.2, alpha=.7, zorder=3,
        )
        lead_rows.append(selected)
    lead_axis.set_xticks(range(4), GROUPS)
    lead_axis.set_ylim(-.025, 1.05)
    lead_axis.set_yticks([0, .25, .50, .75, 1])
    lead_axis.set_xlim(-.6, 3.6)
    lead_axis.set_ylabel('Lead / start-to-knee span',fontsize=7.8)
    lead_axis.tick_params(labelsize=7.3)
    lead_axis.legend(
        handles=[
            Line2D([], [], ls='', marker='o', color=CORAL, label='In stage', ms=3.5),
            Line2D([], [], ls='', marker='o', color=BLUE, label='Outside / unassessed', ms=3.5),
        ],
        ncol=1, frameon=False, fontsize=7.1, loc='lower center',
        bbox_to_anchor=(.5, 1.055), handlelength=.6, handletextpad=.35,
        columnspacing=.7, borderaxespad=0,
    )
    pd.concat(lead_rows).to_csv(export / 'Figure5_b.csv', index=False)

    # Compare both signals for each cell with evaluable matched windows.
    letter(f,'c',129,3)
    good=paired[paired.evaluable].copy()
    assert len(paired)==190 and len(good)==184
    assert (good.relative_amplification>1).sum()==135
    # A separate row shows exact zeros below the log scale.
    left = ax(f, 145, 22, 41, 59)
    scatter_zero = -5.7
    left.set(xlim=(-.5,3.05),ylim=(-6.0,4.5))
    xx=np.linspace(-.5,3.05,100)
    left.fill_between(xx,xx,4.5,color='#ECF2F6',alpha=.8,zorder=0)
    left.plot(xx,xx,color='#8B99A6',lw=.65,ls=(0,(3,2)),zorder=1)
    left.text(.04,.97,'State higher',transform=left.transAxes,ha='left',va='top',fontsize=7.0)
    left.text(.96,.24,'Capacity higher',transform=left.transAxes,ha='right',va='top',fontsize=7.0)
    for ds,color in zip(DATASETS,COLORS):
        r = good[good.dataset == ds]
        yv = np.full(len(r), scatter_zero)
        np.log10(r.state_amplification.to_numpy(),out=yv,where=r.state_amplification.to_numpy()>0)
        left.scatter(np.log10(r.output_amplification),yv,s=7,c=color,alpha=.72,
                     edgecolors='white',linewidths=.25,zorder=3)
    left.set_xticks([0,1,2,3],['1','10',r'$10^2$',r'$10^3$'])
    left.set_yticks([scatter_zero,-4,-2,0,2,4],['0',r'$10^{-4}$',r'$10^{-2}$','1',r'$10^2$',r'$10^4$'])
    left.plot([-.007,.007],[-5.44,-5.29],transform=left.get_yaxis_transform(),color='#34383C',lw=.55,clip_on=False)
    left.plot([-.007,.007],[-5.25,-5.10],transform=left.get_yaxis_transform(),color='#34383C',lw=.55,clip_on=False)
    left.set_xlabel('Capacity variance ratio',fontsize=7.5)
    left.tick_params(labelsize=7.2)
    left.set_ylabel('State variance ratio',fontsize=7.5)
    key = ax(f, 129, 10, 59, 6)
    key.axis('off')
    key.legend(handles=[Line2D([],[],ls='',marker='o',color=color,ms=2.8,label=ds)
                        for ds,color in zip(DATASETS,COLORS)],ncol=5,frameon=False,
               loc='center',fontsize=7.0,handlelength=.4,handletextpad=.22,columnspacing=.36)
    paired.to_csv(export/'Figure5_c_paired_representation.csv',index=False)

    letter(f, 'd', 3, 93)
    letter(f, 'e', 70, 93)
    matrix_axis = ax(f, 89, 116, 97, 33)
    matrix_axis.pcolormesh(
        np.arange(len(matrix_datasets) + 1) - .5, np.arange(6) - .5,
        comparison_matrix.to_numpy(), cmap=ListedColormap(OUTCOME_COLORS),
        norm=BoundaryNorm(np.arange(-.5, 6.5), 6), shading='flat',
    )
    matrix_axis.set_ylim(4.5, -.5)
    matrix_axis.set_yticks(range(5), METHODS, fontsize=7.5)
    matrix_axis.get_yticklabels()[0].set_weight('bold')
    matrix_axis.set_xticks([])
    matrix_axis.tick_params(length=0)
    matrix_axis.spines[:].set_visible(False)
    for boundary in [.5, 1.5, 2.5, 3.5]:
        matrix_axis.axhline(boundary, color='white', lw=.65)
    for dataset in DATASETS:
        indices = np.flatnonzero(matrix_datasets == dataset)
        start, end = min(indices), max(indices)
        matrix_axis.text((start + end) / 2, -.72, dataset, ha='center', fontsize=7.3)
        if start:
            matrix_axis.axvline(start - .5, color='white', lw=1.4)
    matrix_axis.text(
        .5, -.08, 'Individual evaluation cells', transform=matrix_axis.transAxes,
        ha='center', va='top', fontsize=8.0,
    )
    comparison_matrix.to_csv(export / 'Figure5_e_matrix.csv')
    legend_axis = ax(f, 74, 96, 113, 11)
    legend_axis.axis('off')
    outcome_labels = [
        'In stage before knee', 'Outside stage before knee', 'Unassessed before knee',
        'At knee', 'After knee', 'No alarm',
    ]
    legend_axis.legend(
        handles=[Patch(facecolor=color, label=label)
                 for color, label in zip(OUTCOME_COLORS, outcome_labels)],
        ncol=3, loc='center', frameon=False, fontsize=7.1, handlelength=.9,
        handletextpad=.4, columnspacing=1.0, labelspacing=.4,
    )

    battery_comparison = comparison
    rate_axis = ax(f, 24, 116, 41, 33)
    for j, method in enumerate(METHODS):
        scores = battery_comparison.loc[method]
        for offset, metric, color in [
            (-.23, 'warning_rate', BLUE), (.23, 'science_effectiveness', CORAL),
        ]:
            percent = scores[metric] * 100
            rate_axis.barh(j + offset, percent, height=.26, color=color, alpha=.85)
            rate_axis.text(
                percent + 1, j + offset, f'{percent:.1f}', va='center', fontsize=7.0,
                weight='bold' if method == 'Proposed' else 'normal',
            )
    rate_axis.set_ylim(4.5, -.5)
    rate_axis.set_xlim(0, 115)
    rate_axis.set_yticks(range(5), METHODS, fontsize=7.4)
    rate_axis.tick_params(labelsize=7.4)
    rate_axis.get_yticklabels()[0].set_weight('bold')
    rate_axis.spines['left'].set_visible(True)
    rate_axis.set_ylabel('Method', fontsize=7.8, labelpad=3)
    rate_axis.grid(False)
    rate_axis.set_axisbelow(True)
    rate_axis.set_xticks([0, 50, 100])
    rate_axis.set_xlabel('Rate (%)',fontsize=8)
    rate_axis.legend(
        handles=[
            Patch(facecolor=BLUE, label='Warning rate'),
            Patch(facecolor=CORAL, label='Stage effectiveness'),
        ],
        ncol=2, frameon=False, fontsize=7.1, loc='lower center',
        bbox_to_anchor=(.39, 1.16), borderaxespad=0, handlelength=.8,
        handletextpad=.32, columnspacing=.6, labelspacing=.35,
    )
    _export(f, 5, output_dir, enlarge=True)


def _draw_figure6(timing, ab, comparison, case, diagnostics, output_dir, export):
    f = fig(190, 187)
    iron, iron_ratios, recovery_trace, iron_recovery = diagnostics

    pd.DataFrame(iron).to_csv(export / 'Figure6_a_variance.csv', index=False)
    iron_ratios.to_csv(export / 'Figure6_b_variance_ratios.csv', index=False)
    recovery_trace.to_csv(export / 'Figure6_c_recovery_trace.csv', index=False)
    pd.DataFrame(iron_recovery['mean_variances']).to_csv(export / 'Figure6_c_variances.csv', index=False)
    letter(f, 'a', 3, 3)
    a = ax(f, 17, 18, 43, 41)
    clock = iron['clock']
    variance_trace = iron['variance']
    a.plot(clock, np.where(variance_trace > 0, variance_trace, np.nan), color=ROSE, lw=.55)
    for start_key, end_key, color in [
        ('early_start', 'early_end', EARLY),
        ('late_start', 'late_end', NEAR),
    ]:
        lo, hi = float(iron[start_key]), float(iron[end_key])
        a.axvspan(lo, hi, color=color, lw=0)
        a.hlines(np.nanmedian(variance_trace[(clock >= lo) & (clock < hi)]), lo, hi, color='black', lw=1)
    a.axvline(float(iron['transition']), color='black', ls='--', lw=.7)
    a.set_yscale('log')
    a.set_xlim(clock[0], clock[-1])
    a.set_yticks([1e-13, 1e-12])
    a.set_xlabel('Recorded time (s)', fontsize=7.6)
    a.set_ylabel('$V(C_2)$')
    a.xaxis.set_major_locator(ticker.MaxNLocator(3))
    a.yaxis.set_minor_locator(ticker.NullLocator())
    a.legend(
        handles=[
            Patch(facecolor=EARLY, label='Earlier'),
            Patch(facecolor=NEAR, label='Near transition'),
            Line2D([], [], color='black', lw=1, label='Median'),
            Line2D([], [], color='black', ls='--', lw=.7, label='Transition'),
        ],
        loc='lower center', bbox_to_anchor=(.5, 1.07), ncol=2, frameon=False,
        borderaxespad=0, fontsize=7.4, columnspacing=1.2, handlelength=1.3,
    )

    letter(f, 'b', 65, 3)
    a = ax(f, 78, 18, 36, 41)
    for j, window in enumerate([10, 30, 60]):
        subset = iron_ratios[iron_ratios.window == window]
        ratio_group(a, subset.variance_ratio, j, '#C5A5B4')
    a.set_yscale('log')
    a.set_ylim(.2, 12)
    a.set_yticks([.3, 1, 3, 10], ['0.3', '1', '3', '10'])
    a.axhline(1, color=GREY, ls='--', lw=.65)
    a.set_xticks(range(3), ['10', '30', '60'])
    a.set_xlabel('Variance window (points)', fontsize=7.6)
    a.set_ylabel('Near / earlier variance', fontsize=7.6)
    a.yaxis.set_minor_locator(ticker.NullLocator())

    letter(f, 'c', 119, 3)
    a = ax(f, 134, 18, 51, 21)
    a.plot(recovery_trace.clock, recovery_trace.kappa, color=GREY, lw=.55)
    for period, color in [('earlier', EARLY), ('near', NEAR)]:
        lo, hi = iron_recovery['intervals'][period]
        kappa = next(row for row in iron_recovery['recovery_primary_statistics'] if row['period'] == period)
        a.axvspan(lo, hi, color=color, lw=0, zorder=-1)
        a.hlines(kappa['kappa_median'], lo, hi, color='black', lw=1)
    a.axvline(iron_recovery['transition'], color='black', ls='--', lw=.65)
    a.set_yscale('log')
    a.set_yticks([.3, 1, 3], ['0.3', '1', '3'])
    a.yaxis.set_minor_locator(ticker.NullLocator())
    a.set_xlabel('Recorded time (s)', fontsize=7.3)
    a.set_ylabel(r'$\kappa_{\mathrm{eff}}$ (s$^{-1}$)', fontsize=7.5)
    a.xaxis.set_major_locator(ticker.MaxNLocator(3))

    a = ax(f, 134, 51, 51, 11.5)
    a.barh([1, 0], [row['residual_relative'] for row in iron_recovery['mean_variances']],
           color=['#93A6B6', '#D9987F'], height=.55)
    a.set_yticks([1, 0], ['Earlier', 'Near'])
    a.set_xlabel('Residual / earlier raw variance', fontsize=7.2)
    a.set_xticks([0, 1, 2])
    a.spines['left'].set_visible(False)
    a.tick_params(axis='y', length=0)
    a.set_xlim(0, 2)

    p4 = WarningPanels(187)
    snapshot = p4.integrated_row(f, case, 'd', 75, True)
    for step in f.axes:
        if any(t.get_text() in {'1', '2', '3'} for t in step.texts):
            step.lines[0].set_markersize(11)

    # One evaluation row for warning timing, benchmarks and the ablation.
    letter(f, 'e', 3, 133)
    fatigue_timing = timing[timing.dataset == 'Soft_Magnetic_Iron']
    stage_aligned = fatigue_timing[fatigue_timing.science_valid]
    a = ax(f, 15, 146, 11, 30)
    box(a, [fatigue_timing.loc[fatigue_timing.warning, 'normalized_lead']], [0], [CORAL], width=.36, pts=4)
    a.set_xlim(-.55, .55)
    a.set_xticks([0], ['Lead'])
    a.set_ylim(-.025, 1.05)
    a.set_yticks([0, .5, 1])
    a.set_ylabel('Lead / pretransition span', fontsize=7.3)

    a = ax(f, 41.5, 146, 28.5, 30)
    box(a, [stage_aligned.first_episode_post_alarm_coverage, stage_aligned.post_alarm_episode_coverage],
        [0, 1], [CORAL, CORAL], width=.36, pts=4)
    a.set_xlim(-.55, 1.55)
    a.set_xticks([0, 1], ['Alarm\nepisode', 'All\nepisodes'], fontsize=7.1)
    a.set_ylim(-.025, 1.05)
    a.set_yticks([0, .5, 1])
    a.set_ylabel('Post-alarm coverage', fontsize=7.3)

    letter(f, 'f', 68, 133)
    fatigue_comparison = comparison
    methods = METHODS
    a = ax(f, 80, 146, 53, 30)
    method_positions = np.array([0, 1.45, 2.45, 3.45, 4.45])
    for offset, metric, color, label in [
        (-.18, 'warning_rate', BLUE, 'Warning rate'),
        (.18, 'science_effectiveness', CORAL, 'Stage effectiveness'),
    ]:
        a.bar(method_positions + offset, 100 * fatigue_comparison.loc[list(methods), metric],
              width=.34, color=color, label=label, zorder=3)
    a.set_xticks(method_positions, methods, fontsize=7.0)
    a.set_xlim(-.61, 5.04)
    a.set_ylim(0, 108)
    a.set_yticks([0, 50, 100])
    a.set_ylabel('Rate (%)', fontsize=7.3)
    a.legend(loc='lower center', bbox_to_anchor=(.5, 1.045), ncol=1, frameon=False,
             fontsize=7.2, handlelength=1, columnspacing=.7, handletextpad=.4,
             borderaxespad=0)

    letter(f, 'g', 136, 133)
    fatigue_ablation = ab[ab.dataset == 'Soft_Magnetic_Iron'].set_index('group')
    groups = ['G1', 'G2', 'G3', 'G4']
    group_positions = np.array([0, 1, 2.25, 3.25])
    selected = fatigue_ablation.loc[groups]
    outcome_counts = pd.DataFrame({
        'Stage-aligned': selected.science_valid,
        'Other timely': selected.warnings - selected.science_valid,
        'Late / absent': selected.n - selected.warnings,
    })
    a = ax(f, 146, 146, 39, 30)
    left = np.zeros(len(groups))
    for label, color in [('Stage-aligned', CORAL), ('Other timely', BLUE),
                         ('Late / absent', '#DADFE4')]:
        values = outcome_counts[label].to_numpy()
        a.barh(group_positions, values, left=left, height=.63, color=color,
               edgecolor='white', linewidth=.3, label=label)
        for y, start, count in zip(group_positions, left, values):
            if count > 0:
                a.text(start + count / 2, y, str(int(count)), ha='center', va='center',
                       fontsize=9.0, color='#20252B')
        left += values
    a.set_ylim(3.95, -.85)
    a.set_xlim(0, 66)
    a.set_yticks(group_positions, groups, fontsize=7.4)
    a.set_xticks([0, 33, 66])
    a.set_xlabel('Evaluation specimens', fontsize=7.5)
    a.spines['left'].set_visible(False)
    a.tick_params(axis='y', length=0)
    a.legend(loc='lower center', bbox_to_anchor=(.5, 1.045), ncol=2, frameon=False,
             fontsize=6.9, handlelength=.95, columnspacing=.75, handletextpad=.35,
             borderaxespad=0)
    outcome_counts.to_csv(export / 'Figure6_g_ablation_outcomes.csv')

    fatigue_timing.to_csv(export / 'Figure6_warning_records.csv', index=False)
    fatigue_comparison.to_csv(export / 'Figure6_f_comparator_rates.csv')
    _export(f, 6, output_dir, enlarge=True)
    return snapshot


def build_figures_456(output_dir, outcomes=None, histories_by_batch=None):
    """Render Figures 4–6 and export their plotted data from specimen-level inputs."""
    if outcomes is None and histories_by_batch is None:
        outcomes, histories_by_batch = collect()
    elif outcomes is None or histories_by_batch is None:
        raise ValueError('Supply both specimen outcomes and specimen histories.')
    output_dir = Path(output_dir)
    export = output_dir / 'panel_data'
    export.mkdir(parents=True, exist_ok=True)
    evaluation, timing = evaluation_timing(outcomes[outcomes.role.eq('evaluation_60pct')], histories_by_batch)
    batteries = timing[timing.dataset.isin(DATASETS)]
    condition = condition_outcomes(batteries, histories_by_batch)
    paired = matched_variance(batteries, histories_by_batch)
    matrix, matrix_datasets = comparison_matrix(evaluation)
    ablation = rates(evaluation[evaluation.method.isin(GROUPS)], ('dataset', 'method')).rename(
        columns={'method': 'group'})
    comparison = evaluation[evaluation.method.isin(('G4', *METHODS[1:]))].copy()
    comparison['method'] = comparison.method.replace({'G4': 'Proposed'})
    battery_rates = rates(comparison[comparison.dataset.isin(DATASETS)], ('method',)).set_index('method')
    fatigue_rates = rates(comparison[comparison.dataset.eq(FATIGUE)], ('method',)).set_index('method')
    battery_rates.loc[list(METHODS)].to_csv(export / 'Figure5_d_comparator_rates.csv')
    if tuple(battery_rates.loc['Proposed', ['n', 'warnings', 'science_valid']]) != (190, 172, 151):
        raise ValueError('Battery warning counts differ from the manuscript.')
    if tuple(fatigue_rates.loc['Proposed', ['n', 'warnings', 'science_valid']]) != (66, 61, 61):
        raise ValueError('Fatigue warning counts differ from the manuscript.')
    ablation.to_csv(export / 'Figure5_a_ablation_rates.csv', index=False)
    cases = (warning_case('MIT', 'c22', histories_by_batch), warning_case(FATIGUE, '131', histories_by_batch))
    diagnostics = fatigue_diagnostics(outcomes, histories_by_batch)
    with plt.rc_context(_style()):
        battery_snapshot = _draw_figure4(timing, condition, cases[0], output_dir, export)
        _draw_figure5(ablation, evaluation, paired, matrix, matrix_datasets, battery_rates, output_dir, export)
        fatigue_snapshot = _draw_figure6(timing, ablation, fatigue_rates, cases[1], diagnostics, output_dir, export)
    for number, snapshot, case in zip((4, 6), (battery_snapshot, fatigue_snapshot), cases):
        (export / f'Figure{number}_confirmation.json').write_text(
            json.dumps(snapshot, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        pd.DataFrame({key: case['record'][key] for key in (
            'clock', 'raw_variance', 'variance', 'reference', 'gain', 'score',
            'persistence_bound')}).to_csv(export / f'Figure{number}_confirmation_trace.csv', index=False)
    return {number: output_dir / f'Figure_{number}.pdf' for number in (4, 5, 6)}
