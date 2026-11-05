import pathlib,hashlib,json,shutil,sys
root=pathlib.Path(sys.argv[1]).resolve(); review=pathlib.Path(__file__).resolve().parent; name='pr22-ef2b051-20261101-Olwchs.tar.gz'; folder='06_PR22_KV缓存预分配'; expected='befda6337d7ab514790647ee1354fbd612e8e7f6464754bf64f5eb8a78632110'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=root/name; dest=root/folder; target=dest/name
assert source.parent==root and dest.parent==root and target.parent==dest
if not source.is_file() or sha(source)!=expected:raise ValueError('source missing or hash mismatch')
for n,backup in [('索引.md','原索引备份.md'),('文件清单.json','原文件清单备份.json')]:
 if (root/n).read_bytes()!=(review/backup).read_bytes():raise ValueError('index changed since backup')
if target.exists() and sha(target)!=expected:raise FileExistsError(target)
for n in ['简短分析.md','核验清单.json']:
 if (dest/n).exists() and (dest/n).read_bytes()!=(review/n).read_bytes():raise FileExistsError(dest/n)
d=json.loads((root/'文件清单.json').read_text(encoding='utf-8-sig')); rel=folder+'/'+name
if any(x['整理后路径']==rel for x in d['文件']):raise ValueError('entry already exists')
d['文件'].append({'实验日期':'2026-11-01','原文件名':name,'整理后路径':rel,'字节数':source.stat().st_size,'SHA256':expected})
index=(root/'索引.md').read_text(encoding='utf-8-sig').rstrip()+'\n- 2026-11-01：[ '+name+' ]('+rel+')\n'
dest.mkdir(exist_ok=True)
if not target.exists():
 with source.open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)
if sha(target)!=expected or sha(source)!=expected:raise ValueError('copy hash mismatch')
for n in ['简短分析.md','核验清单.json']:
 if not (dest/n).exists():shutil.copyfile(review/n,dest/n)
(root/'索引.md').write_text(index,encoding='utf8');(root/'文件清单.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
# Keep root source as a recoverable copy; no deletion.
print('OK: copied and verified; original retained')
