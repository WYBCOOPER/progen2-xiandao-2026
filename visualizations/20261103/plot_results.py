from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'data.json').read_text(encoding='utf8')); R=D['runs']
fonts={x.name for x in font_manager.fontManager.ttflist};plt.rcParams['font.family']=['Microsoft YaHei' if 'Microsoft YaHei' in fonts else 'SimHei' if 'SimHei' in fonts else 'DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
T=np.array([[x['time_s'] for x in r['result']['loads']] for r in R]); C=np.array([[x['consistency_rate']*100 for x in r['result']['loads']] for r in R]); labels=[r['display_label'] for r in R]; colors=['#dd8452' if r['purpose_group']=='diagnostic' else '#4c78a8' for r in R]
def save(fig,name):fig.savefig(HERE/name,dpi=105,bbox_inches='tight');plt.close(fig)
def plot_all():
 fig,axs=plt.subplots(1,4,figsize=(15,10),sharey=True); y=np.arange(len(R))
 for i,ax in enumerate(axs):
  ax.barh(y,T[:,i],color=colors);ax.set_title(f'负载 {i+1}');ax.set_xlabel('秒');ax.set_xlim(left=0);ax.grid(axis='x',alpha=.2)
 axs[0].set_yticks(y,labels);axs[0].invert_yaxis();fig.suptitle('历史测试耗时（跨日仅展示，不推断加速；橙色为诊断）');fig.tight_layout();save(fig,'history_latency.png')
 fig,ax=plt.subplots(figsize=(9,10)); im=ax.imshow(C,aspect='auto',vmin=0,vmax=100,cmap='YlGn',interpolation='nearest')
 ax.set_yticks(y,labels);ax.set_xticks(range(4),[f'负载 {i+1}' for i in range(4)]);ax.set_title('各次公开测试输出一致率（%）')
 for row in range(len(R)):
  for col in range(4):ax.text(col,row,f'{C[row,col]:.1f}%',ha='center',va='center',fontsize=8,color='white' if C[row,col] >= 70 else 'black')
 fig.colorbar(im,ax=ax,label='一致率（%）');fig.tight_layout();save(fig,'output_consistency.png')
 combo=T[next(i for i,r in enumerate(R) if r['label']=='组合版 同轮对照')];control=T[next(i for i,r in enumerate(R) if r['label']=='V3 监测对照')];reduction=(control.sum()-combo.sum())/control.sum()*100;x=np.arange(4);fig,ax=plt.subplots(figsize=(9,5));bars1=ax.bar(x-.18,control,.36,label='V3 监测对照');bars2=ax.bar(x+.18,combo,.36,label='组合版');ax.bar_label(bars1,fmt='%.4f',padding=3,fontsize=8);ax.bar_label(bars2,fmt='%.4f',padding=3,fontsize=8);ax.set_xticks(x,[f'负载 {i+1}' for i in x]);ax.set_ylim(0,max(control.max(),combo.max())*1.14);ax.set_ylabel('秒');ax.set_title(f'同机各一轮：四项合计降低 {reduction:.2f}%（非稳定性结论）');ax.legend();fig.tight_layout();save(fig,'combo_vs_v3.png')
if __name__=='__main__':plot_all()
