"""Draw the observed evidence and confirmation steps for a single record."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter


class WarningPanels:
    PAGE_W = W = 190.
    INK = '#000000'
    BLUE, BLUE_DARK = '#8DAEC8', '#648DB0'
    CORAL, CORAL_DARK = '#DDA28D', '#C48270'
    GREY, LIGHT, ALARM = '#B6BDC4', '#E1E5E9', '#C47D75'

    def __init__(self, height):
        self.PAGE_H = self.LAYOUT_H = height

    def axes_mm(self, fig, x, y, w, h, owner):
        ax = fig.add_axes([x / self.PAGE_W, 1 - (y + h) / self.PAGE_H, w / self.PAGE_W, h / self.PAGE_H])
        ax._panel_owner = owner
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(pad=2)
        return ax

    def text_mm(self, fig, x, y, text, owner, size=9.0, **kwargs):
        obj = fig.text(x / self.PAGE_W, 1 - y / self.PAGE_H, text, fontsize=size, color=self.INK, va='top', **kwargs)
        obj._panel_owner = owner
        return obj

    def letter(self, fig, owner, x, y):
        obj = self.text_mm(fig, x, y, owner, owner, size=12.5, weight='bold')
        obj.set_gid('panel-letter-' + owner)

    def scalar(self, rec, key):
        return float(np.asarray(rec[key]).item())

    def stage_overlay(self, ax, rec, color, alpha=0.14):
        for (lo, hi) in zip(rec['episode_start'], rec['episode_end']):
            ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0, zorder=0)

    def chronological_lines(self, ax, alarm, transition):
        ax.axvline(transition, color=self.INK, ls=(0, (3, 2)), lw=0.85, zorder=8)
        ax.axvline(alarm, color=self.ALARM, lw=0.9, zorder=8)

    def detector_band(self, fig, x, y, w, h, rec, owner):
        ax = self.axes_mm(fig, x, y, w, h, owner)
        clock = rec['clock']
        alarm = self.scalar(rec, 'alarm_s')
        ax.set_xlim(clock[0], clock[-1])
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.axvspan(clock[0], alarm, ymin=0.24, ymax=0.78, color=self.LIGHT, lw=0)
        ax.axvspan(alarm, clock[-1], ymin=0.24, ymax=0.78, color=self.ALARM, alpha=0.72, lw=0)
        self.chronological_lines(ax, alarm, self.scalar(rec, 'transition_s'))
        ax.text(-0.025, 0.51, 'Alarm', transform=ax.transAxes, ha='right', va='center', fontsize=9.0, clip_on=False)
        ax.text((alarm + clock[-1]) / 2, 0.49, 'on', ha='center', va='center', fontsize=7.8)
        return ax

    def box_points(self, ax, groups, colors, positions=None, log=False, marker_size=5):
        """Actual sample points with deterministic position jitter only."""
        if positions is None:
            positions = np.arange(1, len(groups) + 1)
        for (j, (values, col, pos)) in enumerate(zip(groups, colors, positions)):
            values = np.asarray(values)
            values = values[np.isfinite(values)]
            order = np.argsort(values, kind='stable')
            jitter = 0.18 * np.sin(np.arange(len(values)) * 2.39996323)
            ax.scatter(np.full(len(values), pos) + jitter, values[order], s=marker_size, color=col, alpha=0.55, edgecolors='none', zorder=2)
            if len(values):
                (q1, med, q3) = np.quantile(values, [0.25, 0.5, 0.75])
                ax.add_patch(Rectangle((pos - 0.2, q1), 0.4, q3 - q1, facecolor=col, alpha=0.27, edgecolor=self.INK, lw=0.55, zorder=3))
                ax.plot([pos - 0.2, pos + 0.2], [med, med], color=self.INK, lw=0.95, zorder=4)
                ax.plot([pos, pos], [values.min(), q1], color=col, lw=0.65, zorder=1)
                ax.plot([pos, pos], [q3, values.max()], color=col, lw=0.65, zorder=1)
        if log:
            ax.set_yscale('log')
        ax.set_xlim(min(positions) - 0.55, max(positions) + 0.55)

    def arrow(self, fig, start, end, owner, color='#B6BDC4', lw=0.8):
        obj = FancyArrowPatch((start[0] / self.W, 1 - start[1] / self.LAYOUT_H), (end[0] / self.W, 1 - end[1] / self.LAYOUT_H), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=8, linewidth=lw, color=color, shrinkA=0, shrinkB=0)
        obj._panel_owner = owner
        fig.add_artist(obj)

    def step(self, fig, x, y, n, label, owner):
        ax = self.axes_mm(fig, x, y, 5.2, 5.2, owner)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot(0.5, 0.5, marker='o', ms=9.5, mfc='white', mec=self.INK, mew=0.6)
        ax.text(0.5, 0.5, str(n), ha='center', va='center', fontsize=7.5)
        self.text_mm(fig, x + 6.8, y + 1, label, owner, size=8.4)

    def joint_connector(self, fig, top, owner):
        ax = self.axes_mm(fig, 98, top + 18, 8, 24, owner)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 0.13, 0.62], [0.82, 0.82, 0.5], color=self.GREY, lw=0.8, zorder=1)
        ax.plot([0, 0.13, 0.62], [0.18, 0.18, 0.5], color=self.GREY, lw=0.8, zorder=1)
        shaft = FancyArrowPatch((0.62, 0.5), ((109 - 98) / 8, 0.5), transform=ax.transAxes, arrowstyle='-|>', mutation_scale=8, linewidth=0.8, color=self.GREY, shrinkA=0, shrinkB=0, clip_on=False, zorder=1)
        ax.add_patch(shaft)
        ax.plot(0.62, 0.5, marker='o', ms=5.5, mfc=self.ALARM, mec='white', mew=0.55, linestyle='none', zorder=5)
        ax.text(0.5, 0.99, 'AND', ha='center', va='bottom', fontsize=7.1)

    def decision_gauge(self, fig, x, y, w, h, value, boundary, owner, color, label, readout):
        ax = self.axes_mm(fig, x, y, w, h, owner)
        ax.axis('off')
        ax.set_xlim(0, max(value, boundary) * 1.25)
        ax.set_ylim(-0.45, 1.12)
        ax.plot([0, max(value, boundary) * 1.19], [0.08, 0.08], color=self.LIGHT, lw=0.65)
        ax.plot([0, value], [0.4, 0.4], color=color, lw=2.5, solid_capstyle='butt')
        ax.plot([boundary, boundary], [-0.08, 0.91], color=self.INK, ls=(0, (2, 2)), lw=0.7)
        ax.plot(value, 0.4, marker='o', ms=4.3, color=color, mec='white', mew=0.5, zorder=5)
        ax.text(0, 1.02, label, fontsize=7.7, ha='left', va='bottom')
        ax.text(0, -0.35, readout, fontsize=7.5, ha='left', va='top')
        return ax

    def decision_snapshot(self, fig, case, owner, top, fatigue=False):
        rec = case['record']
        idx = int(self.scalar(rec, 'alarm_index'))
        alarm = self.scalar(rec, 'alarm_s')
        color = self.CORAL if fatigue else self.BLUE
        self.step(fig, 13, top + 8, 1, 'Observed history', owner)
        self.step(fig, 55, top + 8, 2, 'Two conditions', owner)
        self.step(fig, 120, top + 8, 3, 'Confirmed warning', owner)
        self.arrow(fig, (50.4, top + 30), (54.0, top + 30), owner, lw=0.75)
        if not fatigue:
            d = case['config_parameters']['d']
            theta = self.scalar(rec, 'threshold')
            groups = [rec['variance'][idx - 2 * d + 1:idx - d + 1], rec['variance'][idx - d + 1:idx + 1]]
            ax = self.axes_mm(fig, 16, top + 19, 33.7, 21, owner)
            self.box_points(ax, groups, [self.GREY, self.BLUE], log=True, marker_size=4.0)
            ax.axhline(theta, color=self.INK, lw=0.65, ls=(0, (3, 2)), zorder=5)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Previous', 'Recent'], fontsize=7.5)
            ax.tick_params(labelsize=7.0)
            ax.yaxis.set_major_locator(LogLocator(numticks=3))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel('Variance $V(A)$', fontsize=8.1, labelpad=1.4)
            floor = rec['persistence_bound'][idx] / theta
            growth = float(case['config_parameters']['growth'])
            gain = rec['gain'][idx]
            self.decision_gauge(fig, 56, top + 19.5, 40, 8, floor, 1, owner, self.BLUE_DARK, 'Maintained recent level', '%.3f$\\theta$ > $\\theta$' % floor)
            self.decision_gauge(fig, 56, top + 35, 40, 8, gain, growth, owner, self.BLUE_DARK, 'Median change', '%.2f$\\times$ (required %.1f$\\times$)' % (gain, growth))
            self.text_mm(fig, 32.8, top + 47, f'Two {d}-point windows', owner, size=7.6, ha='center')
            checks = {'previous_clock': [float(rec['clock'][idx - 2 * d + 1]), float(rec['clock'][idx - d])], 'recent_clock': [float(rec['clock'][idx - d + 1]), alarm], 'floor_over_threshold': float(floor), 'gain': float(gain), 'required_growth': growth}
        else:
            b = rec['reference'][idx]
            d = case['config_parameters']['d']
            valid = np.flatnonzero(np.isfinite(rec['reference']))
            changes = valid[np.r_[True, np.diff(rec['reference'][valid]) != 0]]
            update = int(changes[changes <= idx][-1])
            length = case['config_parameters']['b']
            prior = rec['variance'][update - length:update] / b
            ax = self.axes_mm(fig, 16, top + 19, 33.7, 21, owner)
            self.box_points(ax, [prior], [self.CORAL], marker_size=3.0)
            ax.axhline(1, color=self.INK, lw=0.6, ls=(0, (3, 2)))
            ax.set_xticks([1])
            ax.set_xticklabels(['Stable history'], fontsize=7.5)
            ax.set_ylim(0, max(3, float(np.nanmax(prior)) * 1.05))
            ax.set_yticks([0, 1, 2, 3])
            ax.set_ylabel('Variance /\nbackground $b$', fontsize=8.1, labelpad=1.4)
            self.text_mm(fig, 32.8, top + 47, 'Background from prior data', owner, size=7.6, ha='center')
            short = rec['short_median'][idx - d + 1:idx + 1] / b
            ax2 = self.axes_mm(fig, 56, top + 20, 40, 7.0, owner)
            ax2.plot(np.arange(d), short, color=self.CORAL_DARK, lw=0.9)
            ax2.scatter(np.arange(d), short, s=4.5, color=self.CORAL_DARK, edgecolors='none')
            ax2.axhline(1.5, color=self.INK, lw=0.65, ls=(0, (2, 2)))
            ax2.set_xlim(-0.5, d - 0.5)
            ax2.set_ylim(1.3, 2.65)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            self.text_mm(fig, 56, top + 16.0, 'Sustained short median', owner, size=7.7)
            self.text_mm(fig, 56, top + 28.4, '30 / 30 decisions > 1.5b', owner, size=7.5)
            long = rec['recent_median_relative_to_reference'][idx]
            self.decision_gauge(fig, 56, top + 37.5, 40, 8, long, 2, owner, self.CORAL_DARK, 'Current long median', '%.2f$b$ > 2$b$' % long)
            checks = {'history_reference_update_clock': float(rec['clock'][update]), 'reference': float(b), 'history_median': float(np.nanmedian(prior)), 'decision_clock': [float(rec['clock'][idx - d + 1]), alarm], 'short_tests_pass': int(np.count_nonzero(short > 1.5)), 'long_over_reference': float(long)}
        self.joint_connector(fig, top, owner)
        return (rec, checks)

    def real_trace(self, fig, rec, case, owner, top, fatigue=False):
        left = 122
        width = 64
        (color, dark) = (self.CORAL, self.CORAL_DARK) if fatigue else (self.BLUE, self.BLUE_DARK)
        clock = rec['clock']
        alarm = self.scalar(rec, 'alarm_s')
        tau = self.scalar(rec, 'transition_s')
        idx = int(self.scalar(rec, 'alarm_index'))
        ax = self.axes_mm(fig, left, top + 19, width, 21, owner)
        self.stage_overlay(ax, rec, color, alpha=0.12)
        if fatigue:
            value = rec['variance'] / rec['reference']
            stat = rec['persistence_relative_to_reference']
            ax.plot(clock, value, color=color, lw=0.55, alpha=0.6, zorder=2)
            ax.plot(clock, stat, color=dark, lw=1.1, zorder=4)
            ax.axhline(1.5, color=self.INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
            ax.set_ylim(0, 5)
            ax.set_yticks([0, 2, 4])
            ax.text(0, 1.035, '$V(C_2)/b$', transform=ax.transAxes, ha='left', va='bottom', fontsize=8.3)
        else:
            value = rec['variance']
            stat = rec['persistence_bound']
            ax.plot(clock, value, color=color, lw=0.75, alpha=0.65, zorder=2)
            ax.plot(clock, stat, color=dark, lw=1.1, zorder=4)
            ax.axhline(self.scalar(rec, 'threshold'), color=self.INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
            ax.set_yscale('log')
            ax.text(0, 1.035, '$V(A)$', transform=ax.transAxes, ha='left', va='bottom', fontsize=8.3)
            ax.yaxis.set_major_locator(LogLocator(numticks=4))
            ax.yaxis.set_minor_formatter(NullFormatter())
        self.chronological_lines(ax, alarm, tau)
        ax.plot(alarm, stat[idx], marker='o', ms=4.8, color=self.ALARM, mec='white', mew=0.6, zorder=8)
        ax.text(alarm, 1.02, 'Confirmed', transform=ax.get_xaxis_transform(), ha='right', fontsize=7.8)
        ax.text(tau, 1.02, '$\\tau$', transform=ax.get_xaxis_transform(), ha='center', fontsize=8.7)
        ax.set_xlim(clock[0], clock[-1])
        ax.tick_params(labelbottom=False)
        d = case['config_parameters']['d']
        windows = [(idx - d + 1, idx, dark)]
        if not fatigue:
            windows.insert(0, (idx - 2 * d + 1, idx - d, self.GREY))
        for (start, end, col) in windows:
            (lo, hi) = (clock[start], clock[end])
            ax.plot([lo, lo, hi, hi], [0.09, 0.04, 0.04, 0.09], transform=ax.get_xaxis_transform(), color=col, lw=0.8, zorder=10)
        band = self.detector_band(fig, left, top + 44, width, 3.1, rec, owner)
        band.axis('on')
        band.spines[['left', 'top', 'right']].set_visible(False)
        band.set_yticks([])
        ticks = MaxNLocator(4).tick_values(clock[0], clock[-1])
        band.set_xticks(ticks[(ticks >= clock[0]) & (ticks <= clock[-1])])
        band.set_xlabel('Recorded time (s)' if fatigue else 'Cycle number', fontsize=8.5, labelpad=1.8)
        return {'dataset': case['dataset'], 'id': str(case['id']), 'alarm_s': alarm, 'transition_s': tau}

    def integrated_row(self, fig, case, owner, top, fatigue=False):
        self.letter(fig, owner, 3, top)
        (color, dark) = (self.CORAL, self.CORAL_DARK) if fatigue else (self.BLUE, self.BLUE_DARK)
        key = self.axes_mm(fig, 13, top + 0.4, 174, 4, owner)
        key.axis('off')
        key.legend(handles=[Line2D([], [], color=color, lw=0.75, label='Variance'), Line2D([], [], color=dark, lw=1.1, label='Persistence statistic'), Patch(facecolor=color, alpha=0.22, label='Stage'), Line2D([], [], color=self.GREY, lw=0.8, marker='|', markersize=3, label='Decision windows'), Line2D([], [], ls='none', marker='o', color=self.ALARM, ms=4, label='Confirmed alarm'), Line2D([], [], color=self.INK, ls=(0, (3, 2)), lw=0.65, label='Transition' if fatigue else 'Knee')], loc='center', ncol=6, frameon=False, fontsize=7.8, handlelength=1.1, columnspacing=1, handletextpad=0.35)
        (rec, checks) = self.decision_snapshot(fig, case, owner, top, fatigue)
        trace = self.real_trace(fig, rec, case, owner, top, fatigue)
        return dict(**trace, snapshot=checks)
