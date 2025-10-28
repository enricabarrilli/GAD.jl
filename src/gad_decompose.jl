export gad_decompose
"""
  This function computes ...
"""
function gad_decompose(F)
    
    X = variables(F)
    d = maxdegree(F)
    n = length(variables(F))-1
    sigma = MultivariateSeries.dual(F,d) 
    M = mult_matrices(sigma)
    
    ms, Z, r = multiplicities(M)
    Zt = transpose(Z)
    Tr = [Zt*M[i]*Z for i in 1:length(M)]

    Subb = local_mult(Tr, ms)
    nPt = size(Subb)[1]
    c   = size(Subb[1])[1]

    Pt = []
    Xi = []
    for i in 1:nPt
        for j in 1:size(Subb[i])[1]
            push!(Xi,tr(Subb[i][j])/size(Subb[i][j],1))
        end
    end
    
    for i in 1:c:length(Xi)
        push!(Pt, Xi[i:i+c-1])
    end
    
    nilx = nil_index(Subb, Pt)

    dg = vcat([nilx[i]-1 for i in 1:length(nilx)]...)

    L=[]

    for i in 1:length(Pt)
        push!(L,dot(X,Pt[i]))
    end
   
    D = DynamicPolynomials.monomials(X,d)

    mlt = [DynamicPolynomials.monomials(X,dg[i]) for i in 1:length(dg)]

    P = vcat([L[i]^(d-dg[i])*mlt[i] for i in 1:length(Pt)]...)

    Vdm = matrixof(P,D)'
    b = matrixof([F],D)'

    
    ws = Vdm\b
   
   if dg == zeros(length(dg))
       return [one(F)*w for w in ws], L,ms
    else
   
       W = []; 

       s = 0
       for i in 1:length(Pt)
           l = length(mlt[i])
           w = dot(mlt[i], ws[s+1:s+l])
           push!(W, dot(mlt[i], ws[s+1:s+l]))
           s+=l
       end
       return W, L, ms
   end
end
