"""Conservative, offline catalog measurements; extraction is NOT compatibility.

Return exact declared labels and their source. Do not infer material, purpose,
physical fit, or medical/skin compatibility from the occurrence of a keyword.
"""
from __future__ import annotations
from decimal import Decimal
import re
import unicodedata


def norm(text):
    # Amazon scrapes prefix many values with U+200E/U+200F. Removing Unicode
    # formatting controls is lossless for these declared scalar/label fields.
    text=''.join(c for c in unicodedata.normalize('NFKC', str(text)) if unicodedata.category(c)!='Cf')
    return ' '.join(text.casefold().split())


def key(text):
    return re.sub(r'[^a-z0-9]', '', norm(text))


def nonempty(v):
    return v is not None and v != '' and v != [] and v != {} and v != 'nan'


def tokens(text):
    return ' '.join(re.findall(r'[a-z0-9]+', norm(text)))


def has_phrase(text, phrase):
    t, p = tokens(text), tokens(phrase)
    return bool(p) and (' '+p+' ') in (' '+t+' ')


def declared_brand(value):
    if not isinstance(value, str): return None
    value = norm(value)
    if value.startswith('brand:'): value = value[6:].strip()
    elif value.startswith('visit the ') and value.endswith(' store'): value = value[10:-6].strip()
    if not value or len(value)>80 or value in {'generic','unknown','n/a','na','unbranded','brand','no brand','none','not applicable'}:
        return None
    return value


def price_cents(value):
    if not isinstance(value,str) or not re.fullmatch(r'\$\s*\d[\d,]*(?:\.\d{1,2})?',value.strip()): return None
    v=Decimal(re.sub(r'[$,\s]','',value))
    return int(v*100) if v>0 else None


def rating(value):
    if type(value) in (int,float):
        value=str(value)
    if not isinstance(value,str): return None
    m=re.fullmatch(r'([0-5](?:\.\d+)?)\s*(?:out of 5 stars)?',norm(value))
    if not m:return None
    val=Decimal(m[1])
    return str(val.normalize()) if 0<val<=5 else None


COLORS={x:x for x in ('black white red blue green pink purple yellow orange brown beige gold silver '
                     'navy ivory cream teal turquoise burgundy maroon lavender coral charcoal bronze copper clear').split()}
COLORS.update({'gray':'grey','grey':'grey','light gray':'light grey','light grey':'light grey',
               'dark gray':'dark grey','dark grey':'dark grey','navy blue':'navy blue',
               'rose gold':'rose gold','off white':'off white','off-white':'off white'})


def color(v):
    return COLORS.get(norm(v)) if isinstance(v,str) else None


MATERIALS={x:x for x in ('wood|solid wood|glass|metal|stainless steel|steel|aluminum|aluminium|silicone|plastic|'
                        'cotton|polyester|leather|velvet|linen|silk|ceramic|porcelain|bamboo|acrylic|nylon|rubber|'
                        'iron|cast iron|wool|faux leather|pu leather|polypropylene|polyethylene|vinyl').split('|')}
MATERIALS['aluminium']='aluminum'
for mat in ('cotton','polyester','linen','silk','wool','nylon'):
    MATERIALS['100% '+mat]='100% '+mat


def material(v):
    return MATERIALS.get(norm(v)) if isinstance(v,str) else None


INTERFACES={'usb type c':'usb-c','usb-c':'usb-c','usb c':'usb-c','usb type-c':'usb-c',
            'usb type a':'usb-a','usb-a':'usb-a','usb a':'usb-a', 'usb type-a':'usb-a',
            'usb':'usb-unspecified','hdmi':'hdmi','lightning':'lightning','displayport':'displayport',
            'micro usb':'micro-usb','micro-usb':'micro-usb','mini usb':'mini-usb','mini-usb':'mini-usb',
            '3.5mm jack':'3.5mm-jack','3.5 mm jack':'3.5mm-jack','3.5 mm':'3.5mm-unspecified',
            'vga':'vga','dvi':'dvi','rca':'rca','ethernet':'ethernet-unspecified',
            'usb 2.0':'usb-2.0-unspecified-connector','usb 3.0':'usb-3.0-unspecified-connector',
            'usb 3.1':'usb-3.1-unspecified-connector','usb 3.2':'usb-3.2-unspecified-connector',
            'thunderbolt':'thunderbolt-unspecified'}


def interface(v):
    return INTERFACES.get(norm(v)) if isinstance(v,str) else None


LENGTH_UNIT={'in':'25.4','inch':'25.4','inches':'25.4','"':'25.4','cm':'10','centimeters':'10',
             'centimetres':'10','mm':'1','millimeters':'1','millimetres':'1'}
LENGTH_RE=re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*(inches|inch|in|cm|centimeters|centimetres|mm|millimeters|millimetres|\")',re.I)


def scalar_length(v):
    if not isinstance(v,str): return None
    m=LENGTH_RE.fullmatch(norm(v))
    if not m:return None
    amount=Decimal(m[1]); unit=m[2].lower()
    if amount<=0:return None
    return {'mm':str((amount*Decimal(LENGTH_UNIT[unit])).normalize()),'unit':unit,'raw':v}


def dimensions(v):
    """Unlabelled ordered product dimensions; not a verified mating/fit geometry."""
    if not isinstance(v,str):return None
    m=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(inches|inch|cm|mm)\s*',norm(v),re.I)
    if not m:return None
    vals=[Decimal(m[i])*Decimal(LENGTH_UNIT[m[4].lower()]) for i in (1,2,3)]
    if any(x<=0 for x in vals):return None
    return {'ordered_mm':[str(x.normalize()) for x in vals],'unit':m[4].lower(),'axis_semantics':'source order only'}


def capacity(v):
    if not isinstance(v,str):return None
    m=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(gb|tb|mb|gib|mib|tib|ml|milliliters?|millilitres?|liters?|litres?|l|'
                   r'fluid ounces?|fl\.?\s*oz\.?|gallons?|quarts?|cups?|ounces?|oz|mah|ah)\s*',norm(v),re.I)
    if not m or Decimal(m[1])<=0:return None
    unit=norm(m[2]); aliases={'milliliter':'ml','milliliters':'ml','millilitre':'ml','millilitres':'ml',
           'liter':'l','liters':'l','litre':'l','litres':'l','fluid ounce':'fl_oz_unspecified_locale',
           'fluid ounces':'fl_oz_unspecified_locale','fl oz':'fl_oz_unspecified_locale','fl. oz.':'fl_oz_unspecified_locale',
           'oz':'oz_mass_or_volume_unspecified','ounce':'oz_mass_or_volume_unspecified','ounces':'oz_mass_or_volume_unspecified'}
    unit=aliases.get(unit,unit)
    # Keep GB vs GiB, fluid vs mass ounces, US vs Imperial unresolved. No guessed conversions.
    return {'amount':str(Decimal(m[1]).normalize()),'unit':unit}


def information(row):
    info=row.get('product_information'); result={}
    if not isinstance(info,dict):return result
    for k,v in info.items():result.setdefault(key(k),[]).append((k,v))
    return result


def visible_blocks(row):
    blocks=[('name',row.get('name','')),('full_description',row.get('full_description',''))]
    bullets=row.get('small_description',[])
    if isinstance(bullets,str):bullets=[bullets]
    if isinstance(bullets,list):blocks.extend((f'small_description[{i}]',v) for i,v in enumerate(bullets) if isinstance(v,str))
    return [(k,v) for k,v in blocks if isinstance(v,str)]


def strict_visible_label(row,kind):
    """Require every explicit same-field declaration to parse and agree.

    A clear short bullet does not excuse an unparsed/conflicting declaration
    elsewhere. This is stricter than counting individual parsable facts.
    """
    if kind not in {'material','color'}:raise ValueError('unsupported_field')
    found=[]
    for path,block in visible_blocks(row):
        if path=='name':continue
        for line in re.split(r'[\n\r;]',block):
            m=re.fullmatch(r'\s*(material|colou?r)\s*:\s*([^\n;]+?)\s*[.]?\s*',line,re.I)
            if not m:continue
            current='material' if m[1].lower()=='material' else 'color'
            if current!=kind:continue
            value=material(m[2]) if kind=='material' else color(m[2])
            if value is None:raise ValueError('unparsed_same_field_declaration')
            found.append({'value':value,'path':path,'raw':line})
    values={e['value'] for e in found}
    if len(values)!=1:raise ValueError('absent_or_conflicting_visible_label')
    return next(iter(values)),found


def facts(row):
    """{bucket:[{value,path,raw,semantic_scope}]} with deterministic finite parsers."""
    out={}
    def add(bucket,value,path,raw,scope):
        if value is not None:
            out.setdefault(bucket,[]).append({'value':value,'path':path,'raw':raw,'semantic_scope':scope})
    b=declared_brand(row.get('brand'));add('brand',b,'brand',row.get('brand'),'declared label; alias equivalence unproven')
    if b and (tokens(row.get('name',''))+' ').startswith(tokens(b)+' '):
        add('brand_visible_prefix',b,'name + brand',row.get('name'),'declared brand corroborated by title prefix')
    c=row.get('product_category')
    if isinstance(c,str) and c.strip():
        parts=[norm(s) for s in re.split(r'\s*[›>]\s*',c) if s.strip()]
        add('category',' > '.join(parts),'product_category',c,'catalog path; default page suppresses field')
        if parts and has_phrase(row.get('name',''),parts[-1]):
            add('category_leaf_visible',parts[-1],'name + product_category',row['name'],'literal leaf label present in title; no inferred ontology')
    p=price_cents(row.get('pricing'));add('price',str(p) if p is not None else None,'pricing',row.get('pricing'),'fixed single-price cents')
    rat=rating(row.get('average_rating'));add('rating',rat,'average_rating',row.get('average_rating'),'displayed rating on (0,5] scale')
    opts=row.get('customization_options'); opts=opts if isinstance(opts,dict) else {}
    all_signatures=[]; selected_colors=[]; color_values=[]; has_color_options=False
    for option_name,entries in opts.items():
        if not isinstance(entries,list) or not entries:continue
        values=[x['value'].strip().replace('/',' | ').lower() for x in entries if isinstance(x,dict) and isinstance(x.get('value'),str) and x['value'].strip()]
        if len(values)==len(entries) and len(values)==len(set(values)):
            all_signatures.append((option_name.lower(),tuple(values)))
            add('option_value_set',str((option_name.lower(),tuple(sorted(values)))),f'customization_options.{option_name}',values,'available values, not completed purchase choice')
        if key(option_name) in {'color','colour'}:
            has_color_options=bool(values)
            for i,x in enumerate(entries):
                if not isinstance(x,dict):continue
                col=color(x.get('value'))
                if col:
                    color_values.append(col)
                    add('color_options',col,f'customization_options.{option_name}[{i}].value',x.get('value'),'available choice; needs selected-options event')
                    if x.get('is_selected') is True:selected_colors.append(col)
        if key(option_name)=='size':
            for i,x in enumerate(entries):
                if not isinstance(x,dict) or not isinstance(x.get('value'),str):continue
                add('size_option_literal',norm(x['value']),f'customization_options.{option_name}[{i}].value',x['value'],'literal option equality only, not physical fit')
    if len(selected_colors)==1:
        add('color_catalog_preselected',selected_colors[0],'customization_options.*.is_selected',selected_colors[0],'catalog scrape default; NOT runtime purchased option')
    info=information(row); blocks=visible_blocks(row)
    material_keys={'material','materialtype','framematerial','fabricstype','fabrictype','primarymaterial'}
    screen_keys={'screensize','standingscreendisplaysize','displaysize','screensizeininches'}
    dim_keys={'productdimensions','itemdimensionslxwxh','itemdimensionslwh'}
    capacity_keys={'capacity','itemvolume','liquidvolume','memorystoragecapacity','digitalstoragecapacity','harddrivesize','flashmemorysize','computermemorysize','batterycapacity'}
    interface_keys={'connectortype','hardwareinterface','interface','interfacetype'}
    for k,entries in info.items():
        for rawk,v in entries:
            path='product_information.'+rawk
            if k in {'color','colour'}:
                col=color(v);add('color_info',col,path,v,'declared field; normally removed from runtime')
                if col and not has_color_options and has_phrase(row.get('name',''),v):
                    add('color_fixed_visible',col,'name + '+path,v,'declared color corroborated by title; no color variants')
            if k in material_keys:
                mat=material(v)
                add('material_info',k+'='+mat if mat else None,path,v,'field-scoped declared material; no property inference')
                if mat and k=='material' and not opts and has_phrase(row.get('name',''),v):
                    add('material_title_corrob',mat,'name + '+path,v,'general declared material in title; multi-material remains possible')
            if k in screen_keys:
                val=scalar_length(v)
                if val:
                    add('screen_size',val['mm'],path,v,'declared screen size; no mounting compatibility')
                    mentions=[scalar_length(m[0]) for m in LENGTH_RE.finditer(row.get('name',''))]
                    if sum(bool(z and z['mm']==val['mm']) for z in mentions)==1:
                        add('screen_size_title_corrob',val['mm'],'name + '+path,v,'matching size mention; title role still needs checking')
                        leaf=norm(row.get('product_category','')).split('›')[-1].strip()
                        if k=='standingscreendisplaysize' and leaf in {'televisions','computer monitors','monitors','traditional laptops','2 in 1 laptops','tablets','all-in-ones'} and not opts and Decimal('25.4')<=Decimal(val['mm'])<=Decimal('3048'):
                            add('screen_size_device_subset',val['mm'],'name + '+path,v,'specific display-device leaf, title agrees, no variants; no fit claim')
            if k in dim_keys:
                val=dimensions(v)
                if val:add('dimensions',str(tuple(val['ordered_mm'])),path,v,'ordered triple, not fit certification')
            if k in capacity_keys:
                val=capacity(v)
                if val:add('capacity',k+'='+val['amount']+' '+val['unit'],path,v,'same field and unit only')
            if k in interface_keys:
                val=interface(v);add('interface_info',val,path,v,'declared connector/interface label; endpoint role/power unproven')
            if k.startswith('compatible') and nonempty(v):
                add('compatible_text',norm(v),path,v,'raw compatibility statement; not a verified relationship')
    # Explicit visible key:value statements can survive clean_product_keys.
    for path,block in blocks:
        if path=='name':continue
        for line in re.split(r'[\n\r;]',block):
            m=re.fullmatch(r'\s*(material|colou?r|capacity|screen size)\s*:\s*([^\n;]+?)\s*[.]?\s*',line,re.I)
            if not m:continue
            field=key(m[1]); raw=m[2].strip(); low=norm(raw)
            if field=='material':
                val=material(raw)
                if val and not opts:add('material_visible_explicit',val,path,raw,'individual exact displayed Material: statement; whole-product agreement still needs strict_visible_label')
            elif field in {'color','colour'}:
                val=color(raw)
                if val and not has_color_options:add('color_fixed_visible',val,path,raw,'explicit displayed Color: label; no color variants')
            elif field=='capacity':
                val=capacity(raw)
                if val:add('capacity_visible_explicit',val['amount']+' '+val['unit'],path,raw,'explicit displayed capacity; compare same units')
            elif field=='screensize':
                val=scalar_length(raw)
                if val:add('screen_size_visible_explicit',val['mm'],path,raw,'explicit displayed Screen size: length')
    # Mentions are deliberately diagnostic only: can refer to supported devices or accessories.
    short=' '.join(v for k,v in blocks if k!='full_description')
    if re.search(r'\b(?:USB[- ]?(?:C|Type[- ]?C)|HDMI|Lightning|DisplayPort)\b',short,re.I):
        add('interface_mentions_only','mention','name/small_description',short[:500],'NOT reliable endpoint compatibility')
    for m in re.finditer(r'\bVESA\s*(?:standard|mount(?:ing)?(?: pattern)?)?\s*:?\s*(\d{2,4})\s*[x×]\s*(\d{2,4})\s*(mm)?',short,re.I):
        if m[3]:add('vesa_explicit_mm',m[1]+'x'+m[2]+'mm','name/small_description',m[0],'explicit pattern only; load/range/role not certified')
    return out


def self_check():
    assert declared_brand('Visit the Acme Store')=='acme'
    assert declared_brand('Generic') is None
    assert price_cents('$12.99')==1299 and price_cents('$10 - $20') is None
    assert rating(0) is None and rating('4.5 out of 5 stars')=='4.5'
    assert color('black and white') is None and color('Gray')=='grey'
    assert material('PU Leather')=='pu leather' and material('Cotton blend') is None
    assert scalar_length('10 inches')['mm']=='254' and scalar_length('up to 10 inches') is None
    assert scalar_length('\u200e10 Inches')['mm']=='254' and capacity('\u200e4 Liters')['unit']=='l'
    assert material('\u200eGlass')=='glass' and color('\u200eGray')=='grey'
    assert dimensions('1 x 2 x 3 cm')['ordered_mm']==['1E+1','2E+1','3E+1']
    assert dimensions('10 x 20') is None and interface('HDMI, USB') is None
    assert capacity('16 ounces')['unit']=='oz_mass_or_volume_unspecified'
    assert capacity('64 GB')['unit']!=capacity('64 GiB')['unit']
    assert strict_visible_label({'small_description':['Material: Glass']},'material')[0]=='glass'
    for lines in (['Material: Glass','Material: Wood'],['Material: Glass','Material: Glass and Metal']):
        try:strict_visible_label({'small_description':lines},'material')
        except ValueError:pass
        else:raise AssertionError('incomplete declaration guard')
    return 17
