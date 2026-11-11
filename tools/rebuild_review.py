"""Recompute numerical tables and plots from archived predictions; never train/infer.
Run: python tools/rebuild_review.py
"""
from pathlib import Path
from zipfile import ZipFile
import argparse, ast, hashlib, io, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES=['normal','benign','malignant']

def table(z,n):return pd.read_csv(io.BytesIO(z.read(n)))
def arrays(z,n):
    with np.load(io.BytesIO(z.read(n)),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def masks(a):
    shape=tuple(int(v) for v in a['mask_shape']);assert shape==(224,224)
    return np.unpackbits(a['mask_bits'],axis=1,count=int(np.prod(shape))).astype(bool)
def align_truth(t,images):
    index={str(v):i for i,v in enumerate(t['images'])}
    assert len(index)==len(images)==len(set(map(str,images)))
    assert set(index)==set(map(str,images))
    order=np.array([index[str(x)] for x in images]);return t['labels'][order],masks(t)[order]
def metrics(y,p,binary,truth):
    d={};normal=y==0;mal=y==2
    if p is not None:
        assert p.shape==(len(y),3) and np.isfinite(p).all()
        assert np.all((p>=0)&(p<=1));np.testing.assert_allclose(p.sum(1),1,atol=1e-5,rtol=0)
        pred=p.argmax(1);cm=np.zeros((3,3),int);np.add.at(cm,(y,pred),1)
        tp=np.diag(cm);f1=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros(3),where=cm.sum(0)+cm.sum(1)>0)
        fn=int(np.count_nonzero(pred[mal]!=2));fp=int(np.count_nonzero(pred[~mal]==2))
        rec=1-fn/mal.sum();spec=1-fp/(~mal).sum()
        d.update(accuracy=float(np.trace(cm)/len(y)),macro_f1=float(f1.mean()),malignant_recall=float(rec),missed_malignant=fn,nonmalignant_predicted_malignant=fp,nonmalignant_specificity=float(spec),malignant_balanced_accuracy=float((rec+spec)/2))
    if binary is not None:
        assert binary.shape==truth.shape
        ta=truth.sum(1);pa=binary.sum(1);inter=np.logical_and(truth,binary).sum(1);pos=ta>0
        dice=np.divide(2*inter,ta+pa,out=np.full(len(y),np.nan),where=pos)
        iou=np.divide(inter,ta+pa-inter,out=np.full(len(y),np.nan),where=pos)
        d.update(positive_dice=float(np.mean(dice[pos])),positive_iou=float(np.mean(iou[pos])),malignant_dice=float(np.mean(dice[mal])),empty_positive_predictions=int(np.count_nonzero(pa[pos]==0)),normal_mean_foreground_percent=float(pa[normal].mean()/truth.shape[1]*100))
    return d

def audit(out):
    checks=[];manifest=ROOT/'data/split_manifest.csv';raw=manifest.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='cafadff2cb11e0eaebe0eb428bba3768a5c179f3702341c3021f314556ef2d3d'
    m=pd.read_csv(manifest);assert len(m)==761 and m.image.is_unique
    assert m.split.value_counts().to_dict()=={'train':499,'test':147,'val':115}
    assert m.groupby('group_id')['split'].nunique().max()==1
    assert m.group_id.nunique()==582
    pd.crosstab(m['split'],m['class_name']).reindex(['train','val','test']).to_csv(out/'split_counts.csv')
    def compare(run,observed,expected,prefix):
        for k,v in observed.items():
            field=prefix+'_'+k
            if field not in expected or pd.isna(expected[field]):continue
            e=float(expected[field]);delta=abs(v-e);ok=delta<=1e-6
            checks.append(dict(run=run,metric=field,expected=e,recomputed=v,absolute_difference=delta,match=bool(ok)))
    A=ROOT/'results/archives'
    with ZipFile(A/'closure_20261102T234653_168026Z.zip') as z:
        truth_test=arrays(z,'old_test_ground_truth.npz')
    with ZipFile(A/'BUSI_FINAL_RESULTS.zip') as z:
        for _,r in table(z,'test_summary.csv').iterrows():
            run=f"{r['variant']}/seed{int(r['seed'])}";a=arrays(z,run+'/test_predictions.npz');y,t=align_truth(truth_test,a['images'])
            compare(run,metrics(y,a['probabilities'],masks(a),t),r,'test')
    with ZipFile(A/'BUSI_Single_vs_Multi_RESULTS.zip') as z:
        truth=arrays(z,'validation_reference.npz')
        for _,r in table(z,'all_selected_checkpoints.csv').iterrows():
            run=f"{r['mode']}_seed{int(r['seed'])}_selected_{r['selection_task']}"
            a=arrays(z,run+'/validation_predictions.npz');y,t=align_truth(truth,a['images'])
            p=a.get('classification');b=masks(a) if 'mask_bits' in a else None
            compare(run,metrics(y,p,b,t),r,'val')
    ck=pd.DataFrame(checks);ck.to_csv(out/'numeric_rechecks.csv',index=False)
    assert ck['match'].all(),ck[~ck['match']].to_dict('records')
    historical=pd.read_csv(ROOT/'results/tables/historical_metric_rechecks.csv')
    flags=historical[historical.status!='MATCH'];flags.to_csv(out/'original_closure_flags.csv',index=False)
    status=json.loads((ROOT/'results/tables/closure_summary.json').read_text())
    result={'numeric_checks':len(ck),'numeric_checks_passed':int(ck.match.sum()),'fixed_split_hash_verified':True,'images':len(m),'similarity_groups':int(m.group_id.nunique()),'original_closure_status':status['status'],'original_closure_flags':len(flags),'weights_or_raw_images_used':False,'training_or_inference_performed':False,'scope':'independent numerical recomputation from archived class probabilities, packed hard masks, and reference masks; does not verify training reproduction or original model bytes'}
    (out/'review_audit.json').write_text(json.dumps(result,indent=2));return result

def summarize(frame,group,metrics):
    rows=[]
    for name,f in frame.groupby(group,sort=False):
        for metric in metrics:
            v=f[metric].dropna();rows.append({'group':name,'metric':metric,'n_seeds':len(v),'mean':v.mean(),'std':v.std(ddof=1)})
    return pd.DataFrame(rows)
def save(fig,folder,name):
    fig.tight_layout(rect=(0,0,1,.88) if fig._suptitle is not None else (0,0,1,1))
    fig.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.12,facecolor='white');plt.close(fig)

def figures(out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    A=ROOT/'results/archives';F=out/'figures';F.mkdir(exist_ok=True)
    hist=pd.read_csv(ROOT/'results/tables/historical_test_summary.csv')
    summary=summarize(hist,'variant',['test_accuracy','test_macro_f1','test_malignant_recall','test_malignant_balanced_accuracy','test_positive_dice','test_positive_iou'])
    summary.to_csv(out/'historical_test_mean_std_tidy.csv',index=False)
    variants=list(hist.variant.unique());labels=['CNN','CNN + T','CNN + AG','CNN + T + AG','Full LPRM v1']
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,label in zip(axs,['test_malignant_balanced_accuracy','test_malignant_recall','test_positive_dice'],['Malignant balanced accuracy','Malignant recall','Positive Dice']):
        rows=summary[summary.metric==metric].set_index('group').loc[variants];ax.bar(range(5),rows['mean'],yerr=rows['std'],capsize=4,color='#396b89');ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_title(label);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Historical test: mean and sample SD across 3 seeds; same 147 images')
    save(fig,F,'historical_test_metrics')
    ci=pd.read_csv(ROOT/'results/tables/historical_paired_group_bootstrap.csv');fig,ax=plt.subplots(figsize=(9,3.3))
    point=ci.cnn_transformer_minus_cnn.to_numpy()*100;lo=ci.lower_95.to_numpy()*100;hi=ci.upper_95.to_numpy()*100
    ax.errorbar(point,range(2),xerr=[point-lo,hi-point],fmt='o',capsize=5);ax.axvline(0,color='#b04a40',ls='--');ax.set_yticks(range(2),['Malignant BA','Positive Dice']);ax.set_xlabel('CNN + T minus CNN (percentage points); recorded 95% group-bootstrap interval');ax.set_title('Recorded paired comparison: 117 similarity groups, 2000 resamples')
    save(fig,F,'historical_primary_bootstrap')
    cls=pd.read_csv(ROOT/'results/tables/core_classification_comparison.csv');seg=pd.read_csv(ROOT/'results/tables/core_segmentation_comparison.csv')
    summarize(cls,'mode',['val_accuracy','val_macro_f1','val_malignant_recall','val_malignant_balanced_accuracy']).to_csv(out/'core_classification_mean_std_tidy.csv',index=False)
    summarize(seg,'mode',['val_positive_dice','val_positive_iou','val_malignant_dice','val_normal_mean_foreground_percent']).to_csv(out/'core_segmentation_mean_std_tidy.csv',index=False)
    for frame,st,metrics,ylims,name in [(cls,'classification_only',['val_accuracy','val_malignant_recall'],[(.7,.95),(.7,.95)],'core_classification'),(seg,'segmentation_only',['val_positive_dice','val_normal_mean_foreground_percent'],[(.55,.63),(0,4.5)],'core_segmentation')]:
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,metric,ylim in zip(axs,metrics,ylims):
            for seed,color in zip([42,43,44],['#31688e','#b56f32','#43846c']):
                values=[float(frame[(frame['mode']==mode)&(frame.seed==seed)][metric].iloc[0]) for mode in [st,'multitask']];ax.plot([0,1],values,'o-',label=f'Seed {seed}',color=color)
            ax.set_xticks([0,1],['Single task','Multi task']);ax.set_title({'val_accuracy':'Three-class accuracy','val_malignant_recall':'Malignant recall','val_positive_dice':'Positive Dice','val_normal_mean_foreground_percent':'False foreground on normal images (%)'}[metric]);ax.set_ylim(*ylim);ax.grid(alpha=.15);ax.legend()
        fig.suptitle('Task-selected validation checkpoints\nSame 115 images across all seeds')
        save(fig,F,name)
    with ZipFile(A/'BUSI_Single_vs_Multi_RESULTS.zip') as z:
        for task,mode,metric in [('classification','classification_only','val_accuracy'),('segmentation','segmentation_only','val_positive_dice')]:
            fig,axs=plt.subplots(1,3,figsize=(13,3.8))
            for ax,seed in zip(axs,[42,43,44]):
                for m,color in [(mode,'#31688e'),('multitask','#b56f32')]:
                    f=table(z,f'{m}_seed{seed}_selected_{task}/training_log.csv');ax.plot(f.epoch,f[metric],label=('ST' if m==mode else 'MTL')+' validation',color=color)
                    if task=='classification':
                        key='accuracy' if m==mode else 'classification_accuracy';ax.plot(f.epoch,f[key],ls=':',alpha=.65,color=color,label=('ST' if m==mode else 'MTL')+' training')
                ax.set_title(f'Seed {seed}');ax.set_xlabel('Epoch');ax.set_ylabel('Accuracy' if task=='classification' else 'Positive hard Dice (validation)');ax.grid(alpha=.18);ax.legend(fontsize=8)
            fig.suptitle('Recorded training logs: fixed 30 epochs\nValidation used for checkpoint selection')
            save(fig,F,'training_'+task)
    with ZipFile(A/'BUSI_FINAL_RESULTS.zip') as z:
        fig,axs=plt.subplots(1,2,figsize=(10,4.2))
        for ax,v in zip(axs,['cnn_only','cnn_transformer']):
            cm=np.zeros((3,3),float)
            for seed in [42,43,44]:
                f=table(z,f'{v}/seed{seed}/test_per_image.csv');np.add.at(cm,(f.label.to_numpy(),f.predicted_label.to_numpy()),1/3)
            pct=100*cm/cm.sum(1,keepdims=True);ax.imshow(pct,vmin=0,vmax=100,cmap='Blues')
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{pct[i,j]:.1f}%',ha='center',va='center',color='white' if pct[i,j]>55 else 'black')
            ax.set_xticks(range(3),NAMES);ax.set_yticks(range(3),NAMES);ax.set_title(v);ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
        fig.suptitle('Mean row-normalized confusion; same 147 images repeated across 3 seeds')
        save(fig,F,'historical_confusion')
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,v in zip(axs,['cnn_only','cnn_transformer']):
            for seed in [42,43,44]:
                f=table(z,f'{v}/seed{seed}/test_per_image.csv');y=(f.label.to_numpy()==2);p=f.p_malignant.to_numpy();order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order];last=np.r_[np.diff(ps)!=0,True];tp=np.cumsum(ys)[last]/y.sum();fp=np.cumsum(~ys)[last]/(~y).sum();ax.plot(np.r_[0,fp],np.r_[0,tp],label=f'Seed {seed}')
            ax.plot([0,1],[0,1],'--',color='gray',alpha=.5);ax.set_title(v);ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate');ax.legend();ax.grid(alpha=.15)
        fig.suptitle('Malignant one-vs-rest ROC; individual fixed models, no ensemble')
        save(fig,F,'historical_roc')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'review_generated');args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    result=audit(args.output);figures(args.output);print(json.dumps(result,indent=2));print('Wrote tables and figures to',args.output)
