"""Render the supplementary figures from the study data and detector results."""
from pathlib import Path
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator, StrMethodFormatter
import numpy as np
import pandas as pd

from battery_models import fit_diagnostics, two_segment_linear_transition
from softiron_models import FREQUENCY_HZ, fit_c2_from_raw, two_segment_linear_transition as fatigue_transition
import manuscript_replay as replay

BLUE, CORAL = '#416C87', '#B96F57'
EARLY, NEAR = '#EAF0F4', '#F4E8DD'
INK, GRID = '#20262C', '#C4CDD4'
DATASETS = ['MIT', 'HUST', 'CityU', 'ISEA', 'Tongji']
BATCHES = ['MIT', 'HUST1', 'HUST2', 'CityU', 'ISEA', 'Tongji']
INPUT = replay.INPUT
FIGURE_INPUT = INPUT.parent / 'publication_figures' / 'si'
FILES = {
    1:'supplementary_battery_capacity_records', 2:'supplementary_state_identification_records',
    3:'supplementary_transition_references', 4:'supplementary_variance_details',
    5:'supplementary_recovery_diagnostics', 6:'supplementary_warning_records',
    8:'supplementary_paired_details', 9:'supplementary_fatigue_observation_records',
    7:'supplementary_warning_persistence', 10:'fatigue_identification',
    11:'fatigue_variance', 12:'fatigue_recovery', 13:'fatigue_warning_records',
    14:'fatigue_persistence',
}
STYLE = {
    'font.family':'Arial', 'font.size':9, 'axes.labelsize':9, 'axes.titlesize':9.4,
    'xtick.labelsize':8.5, 'ytick.labelsize':8.5, 'legend.fontsize':8.5,
    'axes.linewidth':.65, 'axes.edgecolor':INK, 'text.color':INK, 'axes.labelcolor':INK,
    'xtick.color':INK, 'ytick.color':INK, 'axes.unicode_minus':True,
    'pdf.fonttype':42, 'svg.fonttype':'none', 'mathtext.fontset':'custom',
    'mathtext.rm':'Arial', 'mathtext.it':'Arial:italic', 'mathtext.bf':'Arial:bold',
    'path.simplify':False,
}

FATIGUE_EARLY, FATIGUE_NEAR, FATIGUE_GREY = '#DEE8EE', '#F3E4DA', '#B8C4CA'
DASH = (0, (3, 2))


def rolling_variance(series, width):
    variance = np.full(len(series), np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(series, width)
    variance[width - 1:] = windows.var(axis=1, ddof=1)
    return variance


def segments(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)


def figure(height):
    f = plt.figure(figsize=(174/25.4, height/25.4))
    f.mm_height = height
    return f

def axes(f, x, y, w, h):
    a = f.add_axes([x/174, 1-(y+h)/f.mm_height, w/174, h/f.mm_height])
    a.spines[['top','right']].set_visible(False)
    a.tick_params(length=2.6, width=.65, pad=2.6)
    a.xaxis.set_major_locator(MaxNLocator(3))
    a.yaxis.set_major_locator(MaxNLocator(4))
    a.xaxis.get_offset_text().set_fontsize(8.5)
    a.yaxis.get_offset_text().set_fontsize(8.5)
    return a

def canvas(height):
    return figure(height)


def axes_mm(fig, left, bottom, width, height):
    a = axes(fig, left, fig.mm_height - bottom - height, width, height)
    a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_minor_locator(NullLocator())
    return a


def panel_label(fig, letter, left, y, size=12):
    fig.text(left / 174, y / fig.mm_height, letter,
             fontsize=size, fontweight='bold', ha='left', va='center')


def note(f, x, y, s, **kw):
    return f.text(x/174, 1-y/f.mm_height, s, va=kw.pop('va','top'), **kw)

def label(f, x, y, s):
    note(f, x, y, s, fontsize=11.1, weight='bold')

def key(f, handles, ncol, y=0.8):
    f.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51,1-y/f.mm_height),
             ncol=ncol, frameon=False, handlelength=1.5, handletextpad=.48,
             columnspacing=1.2, borderaxespad=.2)

def title(a, value):
    a.set_title(value, loc='left', pad=6, fontsize=9.4)

def span_key(median=True):
    h = [Patch(fc=EARLY,label='Earlier'), Patch(fc=NEAR,label='Near knee')]
    if median:
        h += [Line2D([],[],color=INK,lw=1.15,label='Interval median')]
    return h + [Line2D([],[],color=INK,ls=(0,(3,2)),lw=.8,label='Knee')]

def save(f, n, out):
    for extension in ('pdf', 'svg', 'png'):
        f.savefig(out / f'{FILES[n]}.{extension}', dpi=380, facecolor='white')
    plt.close(f)


def cityu_cells():
    for sid,c in pd.read_csv(INPUT/'CityU_capacity.csv').groupby('cell',sort=True):
        c=c.sort_values('raw_cycle')
        yield sid, c.cycle_since_REF.to_numpy(float), c.SOH_pct.to_numpy(float)/100

def capacity_cohort(batch):
    if batch == 'CityU':
        return list(cityu_cells())
    m = pd.read_csv(INPUT/f'{batch}.csv',dtype={'id':str})
    with np.load(INPUT/f'{batch}.npz') as z:
        return [(sid,z[sid+'_clock'],z[sid+'_capacity']) for sid in sorted(m.id)]

def capacity_axes_label(dataset):
    return 'Capacity (Ah)' if dataset in ('MIT','ISEA') else 'Relative capacity'

def cycle_label(dataset):
    return {'CityU':'Cycle since REF','Tongji':'Cycle number'}.get(dataset,'Observation index')

def plot_capacities(out):
    f=figure(96)
    pos=[(16,10),(74,10),(132,10),(45,57),(103,57)]
    for i,((x,y),ds) in enumerate(zip(pos,DATASETS)):
        a=axes(f,x,y,38,28);label(f,x-13,y-8,'abcde'[i])
        rows = (capacity_cohort('HUST1') + capacity_cohort('HUST2')
                if ds=='HUST' else capacity_cohort(ds))
        for _,t,v in rows:
            a.plot(t,v/v[0] if ds=='HUST' else v,color=BLUE,lw=.43,alpha=.43,
                   rasterized=len(rows)>30)
        title(a,f'{ds}  ($n={len(rows)}$)')
        a.set(xlabel=cycle_label(ds),ylabel=capacity_axes_label(ds))
        a.margins(x=.015,y=.075)
    save(f, 1, out)

def plot_identification(out):
    f=figure(131)
    key(f,[Line2D([],[],color=BLUE,label='Input'),
           Line2D([],[],color=CORAL,ls=(0,(3,2)),label='Reconstruction')],2)
    pos=[(16,17),(74,17),(132,17),(45,77),(103,77)]
    for i,((x,y),ds) in enumerate(zip(pos,DATASETS)):
        batch='HUST1' if ds=='HUST' else ds
        sid,t,v=capacity_cohort(batch)[0]
        q=fit_diagnostics(v,batch)
        a=axes(f,x,y,38,24);b=axes(f,x,y+29,38,13)
        label(f,x-13,y-8,'abcde'[i])
        title(a,f'{ds}: {sid}')
        a.plot(t,v,color=BLUE,lw=.8)
        a.plot(t,q['fitted_capacity'],color=CORAL,lw=.85,ls=(0,(3,2)))
        a.set_ylabel('Capacity (Ah)' if batch in ('MIT','HUST1','ISEA') else 'Relative capacity')
        a.tick_params(labelbottom=False,length=0)
        b.axhline(0,color=GRID,lw=.65)
        b.plot(t,q['residual']*1e3,color=BLUE,lw=.65)
        b.yaxis.set_major_locator(MaxNLocator(2))
        b.set(xlabel=cycle_label(ds),ylabel='Residual\n(mAh)' if batch in ('MIT','HUST1','ISEA') else 'Residual\n'+r'($10^{-3}$)')
        for a1 in (a,b):
            a1.set_xlim(float(t[0]),float(t[-1]))
    save(f, 2, out)

def plot_transitions(out, outcomes):
    f=figure(102)
    key(f,[Line2D([],[],color=BLUE,label='Output record'),
           Line2D([],[],color=CORAL,ls=(0,(3,2)),label='Independent line fits'),
           Line2D([],[],color=INK,ls=(0,(3,2)),label='Fixed reference')],3)
    ids=['b39','2-4','622-Li-D1-25-2','Cell33','CY25-025_1-#4']
    knees = outcomes[outcomes.method.eq('G4')].set_index(['batch', 'id']).transition_s
    with np.load(FIGURE_INPUT / 'battery_knee_examples.npz', allow_pickle=False) as archive:
        examples = dict(archive.items())
    for i,((x,y),ds,sid) in enumerate(zip([(16,16),(74,16),(132,16),(45,62),(103,62)],DATASETS,ids)):
        batch='HUST2' if ds=='HUST' else ds
        if ds=='CityU':
            _,t,v=next(row for row in cityu_cells() if row[0]==sid)
            k=two_segment_linear_transition(t,v)
            assert t[k] == knees.loc[(batch, sid)]
        else:
            t, v = examples[batch + '_clock'], examples[batch + '_capacity']
            k = int(examples[batch + '_split'])
            assert two_segment_linear_transition(t, v) == k
            outcome_id = f'{batch}__{sid}' if batch.startswith('HUST') else sid
            assert t[k] == knees.loc[(batch, outcome_id)]
            if batch == 'HUST2':
                v = v / 1190
        a=axes(f,x,y,38,28);label(f,x-13,y-7.6,'abcde'[i])
        a.plot(t,v,color=BLUE,lw=.9)
        for sl in (slice(0,k),slice(k,None)):
            xx=t[sl];yy=v[sl]
            beta=np.polyfit(xx,yy,1)
            a.plot(xx,np.polyval(beta,xx),color=CORAL,lw=1.05,ls=(0,(3,2)))
        a.axvline(t[k],color=INK,lw=.8,ls=(0,(3,2)))
        a.text(t[k],.95,r'$\tau$ ',transform=a.get_xaxis_transform(),ha='right',va='top')
        title(a,f'{ds}: {sid}')
        a.set(xlabel=cycle_label(ds),ylabel=capacity_axes_label(ds))
        a.set_xlim(float(t[0]),float(t[-1]))
    save(f, 3, out)

def plot_variance(out, histories):
    f=figure(55)
    key(f,span_key()+[Line2D([],[],marker='|',ls='none',color='#788893',label='Zero variance')],5)
    r = histories['MIT']['a11']['record']
    t, z = r['clock'], r['state']
    v = rolling_variance(z, 30)
    tau = r['tau']
    span = t[-1] - t[0]
    ranges = ((tau - .30 * span, tau - .20 * span, EARLY),
              (tau - .05 * span, tau, NEAR))
    for i,(x,q,ylabel) in enumerate(((17,z,r'State $A$'),(105,np.where(v>0,v,np.nan),r'Variance $V(A)$'))):
        a=axes(f,x,17,66,28);label(f,x-14,9,'ab'[i])
        a.plot(t,q,color=BLUE,lw=.85)
        for lo, hi, col in ranges:
            a.axvspan(lo, hi, color=col, lw=0, zorder=0)
            if i:
                median = np.nanmedian(v[(t >= lo) & (t < hi)])
                if median > 0:
                    a.hlines(median, lo, hi, color=INK, lw=1.15)
        a.axvline(r['tau'],color=INK,lw=.8,ls=(0,(3,2)))
        a.set(xlabel='Observation index',ylabel=ylabel,xlim=(t[0],t[-1]))
        if i:
            a.set_yscale('log')
            a.yaxis.set_minor_locator(NullLocator())
            zero=v==0
            a.plot(t[zero],np.full(zero.sum(),.022),'|',color='#788893',ms=2,
                   transform=a.get_xaxis_transform())
    save(f, 4, out)

def linear_recovery_window(clock, state, mode):
    """Return recovery, forcing and innovation estimates for a regular window."""
    steps = np.diff(clock)
    if (not np.isfinite(clock).all() or not np.isfinite(state).all()
            or np.any(steps <= 0)
            or not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-12)):
        return False, np.nan, np.nan, np.nan
    scale = np.std(state)
    if scale == 0:
        return True, np.nan, np.nan, np.nan
    time = (clock - clock.mean()) / (clock[-1] - clock[0])
    values = (state - state.mean()) / scale
    if mode == 'detrended_ar1':
        trend = np.column_stack([np.ones(len(clock)), time])
        values -= trend @ np.linalg.lstsq(trend, values, rcond=None)[0]
        design = np.column_stack([np.ones(len(steps)), values[:-1]])
    else:
        design = np.column_stack([np.ones(len(steps)), time[1:], values[:-1]])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return True, np.nan, np.nan, np.nan
    fit = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    innovation = (values[1:] - design @ fit) * scale
    lag1 = abs(np.corrcoef(innovation[1:], innovation[:-1])[0, 1])
    phi = fit[-1]
    if not 0 < phi < 1:
        return True, np.nan, np.nan, lag1
    kappa = -np.log(phi) / steps[0]
    innovation_variance = np.sum(innovation**2) / (len(steps) - design.shape[1])
    sigma2 = 2 * kappa * innovation_variance / (1 - phi**2)
    return True, kappa, sigma2, lag1


def recovery_sensitivity(system, histories):
    if system == 'battery':
        record = histories['HUST2']['9-6']['record']
        t, tau = record['clock'], record['tau']
        span = t[-1] - t[0]
        intervals = {'earlier': (tau - .3 * span, tau - .2 * span),
                     'near': (tau - .05 * span, tau)}
        windows = [30, 60, 90]
    else:
        record = histories[replay.FATIGUE]['119']['record']
        t, tau = record['clock'], record['tau']
        start = record['output_clock'][0]
        span = tau - start
        intervals = {'earlier': (start + .2 * span, start + .4 * span),
                     'near': (start + .8 * span, tau)}
        windows = [60, 120, 200]
    rows = []
    for window in windows:
        for mode in ('detrended_ar1', 'joint_trend_ar1'):
            for period, (lo, hi) in intervals.items():
                endpoints = np.flatnonzero((t >= lo) & (t < hi))
                results = [
                    linear_recovery_window(t[end-window+1:end+1],
                                           record['state'][end-window+1:end+1], mode)
                    for end in endpoints if end >= window - 1
                ]
                complete, kappa, sigma2, lag1 = np.asarray(results).T
                rows.append(dict(
                    mode=mode, window=window, period=period,
                    valid_fraction_complete=np.isfinite(kappa).sum() / complete.sum(),
                    kappa_median=np.nanmedian(kappa),
                    noise_sigma2_median=np.nanmedian(sigma2),
                    innovation_absrho_median=np.nanmedian(lag1),
                ))
    return pd.DataFrame(rows), windows


def plot_recovery(system, n, out, histories):
    f=figure(62)
    stats, windows = recovery_sensitivity(system, histories)
    ratios = stats.pivot(index=['mode', 'window'], columns='period',
                         values=['kappa_median', 'noise_sigma2_median'])
    ratios = pd.DataFrame({
        'kappa_near_over_earlier': ratios['kappa_median'].near / ratios['kappa_median'].earlier,
        'noise_sigma2_near_over_earlier': (ratios['noise_sigma2_median'].near /
                                        ratios['noise_sigma2_median'].earlier),
    }).reset_index()
    mode_labels={'joint_trend_ar1':'Joint trend + AR(1)','detrended_ar1':'Detrend then AR(1)'}
    markers={'joint_trend_ar1':'o','detrended_ar1':'D'}
    key(f,[Line2D([],[],ls='none',marker=markers[m],color=INK,label=v,ms=4)
           for m,v in mode_labels.items()]
          +[Line2D([],[],ls='none',marker='o',color=INK,mfc='white',label='Earlier',ms=4),
            Line2D([],[],ls='none',marker='o',color=INK,label='Near knee' if system == 'battery' else 'Near transition',ms=4)],4)
    positions=np.arange(3)
    for i,(x,yl) in enumerate(zip([16,74,132],['Near / earlier median','Admissible fits (%)',r'Median innovation $|\rho_1|$'])):
        a=axes(f,x,21,38,30);label(f,x-13,12,'abc'[i])
        a.set_xticks(positions,windows)
        a.set(xlim=(-.4,2.4),ylabel=yl,xlabel='Fitting window (points)' if system=='battery' else 'Fitting window (s)')
        a.axvspan(.76,1.24,color='#F0F3F5',lw=0,zorder=0)
        if i==0:
            a.axhline(1,ls=(0,(3,2)),color='#7D8D98',lw=.7)
            for mode,jitter in zip(markers,(-.055,.055)):
                v=ratios.loc[ratios['mode'].eq(mode)].set_index('window').loc[windows]
                for metric,col,off in (('kappa_near_over_earlier',BLUE,-.12),
                                       ('noise_sigma2_near_over_earlier',CORAL,.12)):
                    a.plot(positions+jitter+off,v[metric],color=col,lw=.8,alpha=.82,
                           marker=markers[mode],ms=4,mew=.7,
                           mfc=col if mode=='joint_trend_ar1' else 'white')
            a.set(ylim=(0,1.6),yticks=[0,.5,1,1.5])
            a.legend(handles=[Line2D([],[],color=BLUE,label=r'$\kappa_{\mathrm{eff}}$'),
                              Line2D([],[],color=CORAL,label=r'$\sigma^2_{\mathrm{eff}}$')],
                     loc='lower center',bbox_to_anchor=(.5,1.015),ncol=2,frameon=False,
                     handlelength=1,handletextpad=.35,columnspacing=.75,borderaxespad=.15)
        else:
            field='valid_fraction_complete' if i==1 else 'innovation_absrho_median'
            factor=100 if i==1 else 1
            for mode,offset in zip(markers,(-.14,.14)):
                for j,w in enumerate(windows):
                    q=stats.loc[stats['mode'].eq(mode)&stats.window.eq(w)].set_index('period')
                    e0,n0=factor*q.loc[['earlier','near'],field].to_numpy()
                    p=j+offset
                    a.plot([p-.045,p+.045],[e0,n0],color=GRID,lw=.85)
                    a.plot(p-.045,e0,marker=markers[mode],color=BLUE,mfc='white',mew=.75,ms=4)
                    a.plot(p+.045,n0,marker=markers[mode],color=BLUE,mfc=BLUE,mew=.75,ms=4)
            if i==1:
                a.set(ylim=(-3,108),yticks=[0,50,100])
            else:
                a.set(ylim=(0,.24),yticks=[0,.1,.2])
    save(f, n, out)

def pairs(out, data):
    methods=['MR','PM','IQRM','TSM']
    d=data.loc[(data.dataset!='Soft_Magnetic_Iron')&data.role.str.startswith('eval')]
    t=d[d.method.isin(['G4',*methods])].pivot(index=['batch','id'],columns='method',values='science_valid')
    assert len(t)==190 and not t.isna().any().any()
    f=figure(47)
    key(f,[Patch(fc=EARLY,label='Proposed only'),Patch(fc=NEAR,label='Comparator only'),
           Patch(fc='#D1DEE5',label='Both')],3)
    for i,method in enumerate(methods):
        x=21+i*39
        a=axes(f,x,14,22,22)
        a.set_title(method,pad=5);label(f,x-7,7,'abcd'[i])
        tab=pd.crosstab(t.G4,t[method]).reindex(index=[False,True],columns=[False,True],fill_value=0)
        colors=[['#F3F4F5',NEAR],[EARLY,'#D1DEE5']]
        for j in range(2):
            for k in range(2):
                a.add_patch(Rectangle((k-.5,j-.5),1,1,fc=colors[j][k],ec='white',lw=1.05))
                a.text(k,j,str(tab.iloc[j,k]),ha='center',va='center',fontsize=10)
        a.set(xlim=(-.5,1.5),ylim=(-.5,1.5))
        a.set_xticks([0,1],['No','Yes'])
        a.set_yticks([0,1],['No','Yes'] if i==0 else [])
        a.tick_params(length=0)
        a.spines[['left','bottom']].set_visible(False)
        if i==0:
            a.set_ylabel('Proposed stage match')
    note(f,90,43,'Comparator stage match',ha='center')
    save(f, 8, out)

def fatigue_panels(out, histories):
    with np.load(FIGURE_INPUT / 'fatigue_129_block.npz', allow_pickle=False) as block:
        raw = np.column_stack([block[k] for k in ('time', 'force', 'displacement', 'magnetic_flux')])
    assert raw.shape == (128, 4) and np.all(np.diff(raw[:, 0]) > 0)
    records = [item['record'] for item in histories[replay.FATIGUE].values()]
    assert len(records) == 110
    example = histories[replay.FATIGUE]['129']['record']
    clock, output = example['output_clock'], example['output']
    mean = raw[:, 2].mean()
    np.testing.assert_allclose(output[clock == 101], [mean], rtol=0, atol=1e-12)
    curves = []
    for record in records:
        trace = record['output']
        curves.append((record['output_clock'], trace - trace[np.isfinite(trace)][0]))
    time_min, time_max = min(t[0] for t, _ in curves), max(t[-1] for t, _ in curves)
    f=figure(95)
    for i,(x,y) in enumerate([(17,11),(104,11),(17,56),(104,56)]):
        a=axes(f,x,y,66,28);label(f,x-14,y-8,'abcd'[i])
        if i<2:
            title(a,['Measured force','Measured displacement'][i])
            a.plot(raw[:,0],raw[:,i+1],color=BLUE,lw=.9,marker='o',ms=1.6,mew=0)
            a.set(xlim=(100,101),xticks=[100,100.25,100.5,100.75,101],
                  ylabel=['Force (kN)','Displacement (mm)'][i])
            a.set_xticklabels(['100','100.25','100.5','100.75','101'])
        elif i==2:
            title(a,'Specimen 129: one-second means')
            a.plot(clock,output,color=CORAL,lw=.9)
            a.scatter([101],[mean],s=23,color=BLUE,edgecolors='white',linewidths=.5,zorder=5)
            a.annotate('100–101 s bin',xy=(101,mean),xytext=(.15,.70),textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='-',color=INK,lw=.65),ha='left',va='center',fontsize=8.5)
            a.set_ylabel('Mean displacement (mm)')
        else:
            title(a,'All 110 retained records')
            for t,v in curves:
                a.plot(t,v,color=CORAL,lw=.42,alpha=.53,rasterized=True)
            a.axhline(0,color=GRID,lw=.65)
            a.set_xscale('log');a.xaxis.set_minor_locator(NullLocator())
            a.set(xlim=(time_min / 1.15, time_max * 1.15),
                  ylabel='Displacement change (mm)')
        a.set_xlabel('Recorded time (s)')
    save(f, 9, out)

    rec = reconstruct_fatigue(raw)
    f=figure(54)
    a=axes(f,19,13,62,30);b=axes(f,106,13,64,30)
    label(f,4,4,'a');label(f,91,4,'b')
    lim=np.array([min(rec['output'].min(),rec['reconstruction'].min()),
                  max(rec['output'].max(),rec['reconstruction'].max())])
    lim=lim+np.array([-1,1])*np.ptp(lim)*.06
    a.plot(lim,lim,ls=(0,(3,2)),color='#82929E',lw=.8)
    a.scatter(rec['output'],rec['reconstruction'],s=10,color=CORAL,alpha=.8,lw=0)
    a.set(xlim=lim,ylim=lim,xlabel='Observed displacement (mm)',ylabel='Reconstructed (mm)')
    b.axhline(0,color=GRID,lw=.7)
    b.plot(rec['time'],1e3*rec['residual'],color=BLUE,lw=.8)
    b.set(xlim=(0,1),xlabel='Elapsed time (s)',ylabel=r'Residual ($\mu$m)')
    save(f, 10, out)

def battery_atlas(out, data, histories):
    f=figure(146)
    pale='#BFCAD2'
    legend=[
        Line2D([],[],color=pale,lw=1.4,label='Finite score through knee'),
        Line2D([],[],color=BLUE,lw=1.5,label='Certified stage'),
        Line2D([],[],color=INK,ls=(0,(3,2)),lw=.8,label='Knee'),
        Line2D([],[],marker='o',color=BLUE,ls='none',ms=4,label='Stage-matched alarm'),
        Line2D([],[],marker='o',mfc='white',color=BLUE,ls='none',ms=4,label='Other timely alarm'),
        Line2D([],[],marker='>',color=CORAL,ls='none',ms=4,label='Late alarm'),
        Line2D([],[],marker='x',color=INK,ls='none',ms=4,label='No alarm'),
    ]
    key(f,legend,4)
    src = data[data.method.eq('G4')].set_index(['batch', 'id'])
    plotted = {d: [] for d in DATASETS}
    for batch in BATCHES:
        dataset = 'HUST' if batch.startswith('HUST') else batch
        for sid, history in histories[batch].items():
            outcome_id = f'{batch}__{sid}' if batch.startswith('HUST') else sid
            result = src.loc[(batch, outcome_id)]
            clock = history['record']['clock']
            tau = result.transition_s
            progress = (clock - clock[0]) / (tau - clock[0])
            active = np.isfinite(history['score']) & (clock <= tau)
            finite = [(progress[lo], progress[hi]) for lo, hi in segments(active)]
            annotated = history['annotation']['intervals']
            certified = [((q['start_s'] - clock[0]) / (tau - clock[0]),
                          (q['end_s'] - clock[0]) / (tau - clock[0])) for q in annotated]
            alarm = (result.alarm_s - clock[0]) / (tau - clock[0])
            plotted[dataset].append((result.role, batch, sid, finite, certified,
                                     alarm, bool(result.science_valid)))
    for i,((x,y),ds) in enumerate(zip([(14,24),(71,24),(128,24),(42,88),(100,88)],DATASETS)):
        a=axes(f,x,y,42,45);label(f,x-11,y-8,'abcde'[i])
        rs=sorted(plotted[ds],key=lambda r:r[:3]);N=len(rs)
        C=sum(r[0].startswith('cal') for r in rs)
        lim=max(1.12,max(r[5] for r in rs if np.isfinite(r[5]))+.06)
        none=lim+.10
        a.axvspan(lim+.035,none+.06,color='#F2F5F6',lw=0,zorder=0)
        a.axvline(1,color=INK,ls=(0,(3,2)),lw=.7,zorder=0)
        a.axhline(C+.5,color='#82929E',ls=(0,(2,2)),lw=.7,zorder=0)
        for j,(_,batch,sid,fins,segs,alarm,match) in enumerate(rs,1):
            for lo,hi in fins:
                a.plot([lo,hi],[j,j],color=pale,lw=.65,solid_capstyle='butt',zorder=1)
            for lo,hi in segs:
                a.plot([lo,hi],[j,j],color=BLUE,lw=.95,solid_capstyle='butt',zorder=2)
            if not np.isfinite(alarm):
                a.plot(none,j,'x',color=INK,ms=2.7,mew=.7)
            elif alarm>1:
                a.plot(alarm,j,'>',color=CORAL,ms=3.1,mew=0)
            else:
                a.plot(alarm,j,'o',color=BLUE,mfc=BLUE if match else 'white',ms=2.5,mew=.5)
        a.set(xlim=(-.025,none+.065),ylim=(N+1,0),xticks=[0,.5,1],yticks=[1,C,N])
        title(a,f'{ds}  ($n={N}$)')
        if i in [0,3]:
            a.set_ylabel('Specimen rank')
        a.set_xlabel('Aligned progress')
    save(f, 6, out)

def fatigue_variance(out, histories):
    record_129 = histories[replay.FATIGUE]['129']['record']
    record_134 = histories[replay.FATIGUE]['134']['record']
    x = record_129['transition_fit_clock']
    y = record_129['transition_fit_output']
    fit = fatigue_transition(x, y)
    split = int(fit['split_index'])
    assert fit['tau'] == record_129['tau'] == 1006.0
    t = record_134['clock']
    z = record_134['state'] * 1e6
    start = record_134['transition_fit_clock'][0]
    tau = record_134['tau']
    span = tau - start
    variance = rolling_variance(record_134['state'], 10)
    intervals = ((start + .2 * span, start + .4 * span, FATIGUE_EARLY),
                 (start + .8 * span, tau, FATIGUE_NEAR))
    medians = [np.nanmedian(variance[(t >= lo) & (t < hi)]) for lo, hi, _ in intervals]
    ratio = medians[1] / medians[0]

    fig = canvas(60)
    a = axes_mm(fig, 17, 13, 39, 32)
    b = axes_mm(fig, 75, 13, 39, 32)
    c = axes_mm(fig, 134, 13, 39, 32)
    for label, ax, title, x_left in [
        ("a", a, "Specimen 129", 1.3),
        ("b", b, "Specimen 134", 61),
        ("c", c, "Specimen 134", 119),
    ]:
        panel_label(fig, label, x_left, 47.6)
        ax.set_title(title, loc="left", pad=4)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    a.plot(x, y, color=BLUE, lw=0.78)
    for lo, hi in [(0, split), (split, len(x))]:
        xx, yy = x[lo:hi], y[lo:hi]
        xx_centered = xx - xx.mean()
        slope = np.sum(xx_centered * (yy - yy.mean())) / np.sum(xx_centered ** 2)
        a.plot(xx, yy.mean() + slope * xx_centered, color=CORAL, lw=1.0, ls=DASH)
    a.axvline(record_129["tau"], color=INK, lw=0.75, ls=DASH)
    a.set_xlim(0, 1120)
    a.set_xticks([0, 400, 800])
    a.set_yticks([.25, .5, .75])
    a.set_ylabel("Mean displacement (mm)", labelpad=4)

    for ax in [b, c]:
        for lo, hi, color in intervals:
            ax.axvspan(lo, hi, color=color, lw=0, zorder=0)
        ax.axvline(record_134["tau"], color=INK, lw=0.75, ls=DASH)
        ax.set_xlim(0, 4280)
        ax.set_xticks([0, 1500, 3000])
    b.plot(t, z, color=BLUE, lw=0.48)
    b.set_yticks([2.5, 5, 7.5])
    b.set_ylabel(r"$C_2$ ($10^{-6}$ mm kN$^{-2}$)", labelpad=4)
    c.plot(t, np.where(variance > 0, variance, np.nan), color=BLUE, lw=0.49)
    c.set_yscale("log")
    c.yaxis.set_major_locator(FixedLocator([1e-13, 1e-12]))
    c.set_ylabel(r"Var($C_2$) (mm$^2$ kN$^{-4}$)", labelpad=4)
    for (lo, hi, _), median in zip(intervals, medians):
        c.hlines(median, lo, hi, color=INK, lw=1.2, zorder=4)
    c.text(.985, 1.076, f"{ratio:.2f}×", transform=c.transAxes,
           ha="right", va="center", fontsize=8.8, color=BLUE)
    fig.text(.55, 0.05, "Recorded time (s)", ha="center", va="bottom", fontsize=9)
    fig.legend(handles=[
        Line2D([], [], color=BLUE, lw=.85, label="Observed series"),
        Line2D([], [], color=CORAL, lw=1.0, ls=DASH, label="Segment fits"),
        Patch(facecolor=FATIGUE_EARLY, label="Earlier"),
        Patch(facecolor=FATIGUE_NEAR, label="Near transition"),
        Line2D([], [], color=INK, lw=1.2, label="Interval median"),
        Line2D([], [], color=INK, lw=.75, ls=DASH, label="Output reference"),
    ], ncol=6, loc="upper center", bbox_to_anchor=(.5, .995), fontsize=8.8,
       frameon=False, handlelength=1.45, handletextpad=.43, columnspacing=1.05)
    save(fig, 11, out)


def fatigue_warning_records(out, outcomes, histories):
    histories = histories[replay.FATIGUE]
    results = outcomes[outcomes.dataset.eq(replay.FATIGUE) & outcomes.method.eq('G4')].set_index('id')
    assert len(histories) == len(results) == 110

    fig = canvas(95)
    axes = [axes_mm(fig, 15, 13, 68, 67), axes_mm(fig, 103, 13, 68, 67)]
    roles = (("calibration_40pct", "Calibration"), ("evaluation_60pct", "Evaluation"))
    for ax, (role, title), letter, left in zip(axes, roles, "ab", (2, 90)):
        ordered = sorted(results.index[results.role.eq(role)], key=int)
        n = len(ordered)
        rank_start = 1 if role == "calibration_40pct" else 45
        ranks = range(rank_start, rank_start + n)
        available, certified = [], []
        stage_aligned, other, late, missed = [], [], [], []
        status = Counter()
        for rank, sid in zip(ranks, ordered):
            h = histories[sid]
            clock = h['record']['clock']
            result = results.loc[sid]
            tau, origin = result.transition_s, clock[0]
            alarm = None if pd.isna(result.alarm_s) else result.alarm_s
            mask = np.isfinite(h['score']) & (clock < tau)
            progress = (clock - origin) / (tau - origin)
            for lo, hi in segments(mask):
                available.append([(progress[lo], rank), (progress[hi], rank)])
            for episode in h['annotation']['intervals']:
                lo = (episode['start_s'] - origin) / (tau - origin)
                hi = (episode['end_s'] - origin) / (tau - origin)
                certified.append([(lo, rank), (hi, rank)])
            is_stage_aligned = bool(result.science_valid)
            if alarm is None:
                missed.append((1.16, rank))
                status["no_alarm"] += 1
            else:
                normalized_alarm = (alarm - origin) / (tau - origin)
                point = normalized_alarm, rank
                if alarm > tau:
                    late.append(point)
                    status["late_alarm"] += 1
                elif is_stage_aligned:
                    stage_aligned.append(point)
                    status["stage_aligned_alarm"] += 1
                else:
                    other.append(point)
                    status["other_timely_alarm"] += 1
        assert sum(status.values()) == n
        if role == "evaluation_60pct":
            assert status == {"stage_aligned_alarm": 61, "late_alarm": 1, "no_alarm": 4}

        ax.axvspan(1.095, 1.225, color="#F2F4F5", lw=0, zorder=-1)
        ax.add_collection(LineCollection(available, colors=FATIGUE_GREY, linewidths=.59,
                                         capstyle="butt", zorder=1))
        ax.add_collection(LineCollection(certified, colors=BLUE, linewidths=.83,
                                         capstyle="butt", zorder=2))
        ax.axvline(1, color=INK, lw=.8, ls=DASH, zorder=0)
        for marks, mark, color, size, stroke in [
            (stage_aligned, "o", BLUE, 2.7, .45),
            (other, "o", BLUE, 2.7, .45),
            (late, ">", CORAL, 3.8, .55),
            (missed, "x", INK, 3.2, .76),
        ]:
            if marks:
                ax.plot(*zip(*marks), marker=mark, ls="none", color=color, ms=size,
                        markeredgewidth=stroke, markerfacecolor="white" if marks is other else color, zorder=4)
        ax.set_xlim(-.015, 1.245)
        ax.set_ylim(rank_start+n-.35, rank_start-.65)
        ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
        ax.set_yticks([rank_start, rank_start+n//2, rank_start+n-1])
        ax.set_title(rf"{title} ($n={n}$)", loc="left", pad=6)
        ax.text(1.16, 1.034, "No alarm", transform=ax.get_xaxis_transform(),
                va="bottom", ha="center", fontsize=8.8)
        panel_label(fig, letter, left, 82.1)
    fig.text(0.025, 0.49, "Record rank", rotation=90, ha="center", va="center", fontsize=9)
    fig.text(.55, .033, "Display-aligned progress", ha="center", va="bottom", fontsize=9)
    fig.legend(handles=[
        Line2D([], [], color=FATIGUE_GREY, lw=1, label="Decision available"),
        Line2D([], [], color=BLUE, lw=1, label="Certified stage"),
        Line2D([], [], color=INK, lw=.8, ls=DASH, label="Output transition"),
        Line2D([], [], marker="o", ls="none", color=BLUE, ms=3.4, label="Stage-aligned alarm"),
        Line2D([], [], marker=">", ls="none", color=CORAL, ms=4.1, label="Late alarm"),
    ], ncol=5, loc="upper center", bbox_to_anchor=(.5, .995), fontsize=8.8,
       frameon=False, handlelength=1.6, handletextpad=.45, columnspacing=1.2)
    save(fig, 13, out)



def draw_duration(ax, cohort, unit):
    span = cohort.remaining_ticks.to_numpy()
    episode = cohort.matched_post_alarm_ticks.to_numpy()
    recurrence = cohort.recurrence_count.to_numpy() > 0
    boundary = cohort.reaches_last_pretransition.to_numpy(dtype=bool)
    colors = np.where(recurrence, CORAL, BLUE)
    upper = span.max() * 1.27
    ax.loglog([0.8, upper], [0.8, upper], dashes=(3, 2), lw=0.85,
              color=GRID, zorder=0)
    for at_boundary, marker, size in [(False, "o", 16), (True, "^", 24)]:
        select = boundary == at_boundary
        ax.scatter(span[select], episode[select], s=size, marker=marker,
                   c=colors[select], alpha=0.90, linewidth=0.25,
                   edgecolors="white", zorder=3 if at_boundary else 4)

    reference = "knee" if unit == "observations" else "transition"
    ax.set(xlim=(0.8, upper), ylim=(0.8, upper),
           xlabel=f"Alarm to {reference} ({unit})",
           ylabel=f"Containing episode after alarm\n({unit})")
    largest_power = int(np.floor(np.log10(upper)))
    powers = np.arange(largest_power + 1)
    # Use the same decade locations on both axes to make equality easy to read.
    powers = powers if largest_power <= 4 else powers[::2]
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(FixedLocator(10.0 ** powers))
        axis.set_minor_locator(NullLocator())
    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.0, color=BLUE,
               label="No later episode"),
        Line2D([], [], marker="o", ls="none", ms=4.0, color=CORAL,
               label="Later episodes"),
        Line2D([], [], marker="^", ls="none", ms=4.6, color=INK,
               label="Reaches observed boundary"),
    ]
    # Matplotlib lays out columns first, keeping the two episode classes adjacent.
    ax.legend(handles=[handles[0], handles[2], handles[1]],
              loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2,
              frameon=False, borderpad=0, borderaxespad=0,
              columnspacing=1.05, handlelength=0.85, handletextpad=0.45,
              labelspacing=0.33)


def draw_coverage(ax, cohort):
    initial = 100 * cohort.first_episode_post_alarm_coverage.to_numpy()
    all_episodes = 100 * cohort.post_alarm_episode_coverage.to_numpy()
    thresholds = np.unique(np.r_[0, initial, all_episodes, 100])
    initial_y = 100 * (initial[:, None] >= thresholds).mean(axis=0)
    all_y = 100 * (all_episodes[:, None] >= thresholds).mean(axis=0)
    ax.fill_between(thresholds, initial_y, all_y, step="pre", color=CORAL,
                    alpha=0.15, lw=0, zorder=1)
    ax.step(thresholds, all_y, where="pre", color=CORAL, lw=1.5,
            label="All certified episodes", zorder=2)
    ax.step(thresholds, initial_y, where="pre", color=BLUE, lw=1.35,
            dashes=(3, 1.7), label="Alarm-containing episode", zorder=3)
    ax.set(xlim=(-1.5, 101.5), ylim=(0, 104),
           xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50, 75, 100],
           xlabel="Post-alarm stage coverage (%)",
           ylabel="Records reaching\ncoverage threshold (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, alpha=0.35, linewidth=0.5)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles[::-1], loc="lower left", bbox_to_anchor=(0, 1.03),
              frameon=False, borderpad=0, borderaxespad=0, handlelength=1.65,
              handletextpad=0.5, labelspacing=0.33)


def reconstruct_fatigue(raw):
    time, force, displacement = raw[:, :3].T
    phase = 2 * np.pi * FREQUENCY_HZ * (time - 100)
    drift = time - time.mean()
    nuisance = np.column_stack([np.ones(time.size), drift])
    harmonic = np.column_stack(
        [nuisance] + [component for order in (1, 2, 3)
                      for component in (np.cos(order * phase), np.sin(order * phase))]
    )
    beta = np.linalg.lstsq(harmonic, force, rcond=None)[0]
    quadrature = beta[2] * np.sin(phase) - beta[3] * np.cos(phase)
    centered_force = force - force.mean()
    design = np.column_stack([nuisance, centered_force, quadrature,
                              centered_force**2, centered_force**3])
    coefficients = np.linalg.lstsq(design, displacement, rcond=None)[0]
    fitted = fit_c2_from_raw(raw, identification_seconds=1, completed_until=101)
    np.testing.assert_array_equal(fitted['clock'], [101])
    np.testing.assert_allclose(fitted['coefficients'][0][:4], coefficients[2:],
                               rtol=1e-6, atol=1e-12)
    reconstruction = design @ coefficients
    return dict(time=time - 100, output=displacement, reconstruction=reconstruction,
                residual=displacement - reconstruction)


def persistence_data(outcomes, histories):
    selected = outcomes[outcomes.method.eq('G4') & outcomes.role.eq('evaluation_60pct')
                        & outcomes.warning & outcomes.science_valid]
    rows = []
    for row in selected.itertuples():
        sid = row.id.split('__', 1)[1] if row.batch.startswith('HUST') else row.id
        history = histories[row.batch][sid]
        episodes = history['annotation']['intervals']
        clock = history['record']['clock']
        post = (clock >= row.alarm_s) & (clock <= row.transition_s)
        containing = next(ep for ep in episodes if ep['start_s'] <= row.alarm_s <= ep['end_s'])
        first = (clock <= containing['end_s']) & post
        all_episodes = np.zeros_like(post)
        for episode in episodes:
            all_episodes |= (clock >= episode['start_s']) & (clock <= episode['end_s'])
        rows.append(dict(
            dataset=row.dataset, remaining_ticks=post.sum(), matched_post_alarm_ticks=first.sum(),
            recurrence_count=sum(ep['start_s'] > containing['end_s'] for ep in episodes),
            reaches_last_pretransition=containing['end_s'] == clock[post][-1],
            first_episode_post_alarm_coverage=first.sum() / post.sum(),
            post_alarm_episode_coverage=(all_episodes & post).sum() / post.sum(),
        ))
    return pd.DataFrame(rows)


def persistence_plot(out, cohort, fatigue):
    n = len(cohort)
    expected = (61, 46, 6) if fatigue else (151, 9, 125)
    recurrence = int(cohort.recurrence_count.gt(0).sum())
    assert (n, recurrence, int(cohort.reaches_last_pretransition.sum())) == expected
    f = figure(66)
    a = axes_mm(f, 16.8, 11, 63.7, 39)
    b = axes_mm(f, 103, 11, 67.4, 39)
    label(f, 2.5, 1.5, 'a')
    label(f, 88.5, 1.5, 'b')
    name = 'Fatigue specimens' if fatigue else 'Battery cells'
    note(f, 16.8, 2, f'{name} ($n = {n}$)', fontsize=9.2)
    note(f, 103, 2, f'Later episodes: {recurrence} of {n}', fontsize=9.2)
    draw_duration(a, cohort, 's' if fatigue else 'observations')
    draw_coverage(b, cohort)
    save(f, 14 if fatigue else 7, out)


def plot_supplementary_figures(output_dir, outcomes, histories_by_batch):
    """Generate Figures S1–S14 and return the PDF paths in figure order."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    histories = histories_by_batch
    cohort = persistence_data(outcomes, histories)
    mechanical = cohort.dataset.eq(replay.FATIGUE)
    style = STYLE.copy()
    try:
        findfont('Arial', fallback_to_default=False)
    except ValueError:
        style.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
    with mpl.rc_context(style):
        plot_capacities(out)
        plot_identification(out)
        plot_transitions(out, outcomes)
        plot_variance(out, histories)
        plot_recovery('battery', 5, out, histories)
        battery_atlas(out, outcomes, histories)
        persistence_plot(out, cohort.loc[~mechanical], fatigue=False)
        pairs(out, outcomes)
        fatigue_panels(out, histories)
        fatigue_variance(out, histories)
        plot_recovery('softiron', 12, out, histories)
        fatigue_warning_records(out, outcomes, histories)
        persistence_plot(out, cohort.loc[mechanical], fatigue=True)
    return tuple(out / f'{FILES[i]}.pdf' for i in range(1, 15))
