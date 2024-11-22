def print_detail_info(DF):
        print(DF.info())
        print(DF.head(5))
        print('-'*30,'NaN 행','-'*30)
        print(DF[DF.isna().any(axis=1)])
        print('-'*30,'중복 행','-'*30)
        print(DF[DF.duplicated()])

def shape_view(*DFs):
    for idx,DF in enumerate(DFs):
        print(f'DF{idx} : [shape: {DF.shape}], [ndim: {DF.ndim}D]')

def add_code(SR):
        irisDF_var = SR.unique().tolist()
        return SR.map(lambda x:irisDF_var.index(x)+1)