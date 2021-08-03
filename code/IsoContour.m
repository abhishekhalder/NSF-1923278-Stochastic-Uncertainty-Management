function [Q,iv,VT,H]=IsoContour(TR,F,iv,vis)
% Retrieve line segments comprising level-sets of a scalar field defined at
% the vertices of triangular surface mesh.
%
% INPUT:
%   - TR    : input surface mesh represented as an object of 'TriRep' 
%             class, 'triangulation' class, or a cell such that TR={Tri,X},
%             where Tri is an M-by-3 array of faces and X is an N-by-3 
%             array of vertex coordinates. 
%   - F     : N-by-1 array specifiying values of the scalar field at the 
%             mesh vertices. 
%   - iv    : 1-by-K vector specifying either the values OR number 
%             level-sets. In case of the latter, level-sets will be 
%             distributed uniformly between the extremal values of F.
%             To get a level-set at one specific value of F (e.g., Fs), 
%             specify iv as Fs*[1 1];
%   - vis   : axes handle (or logical value) indicating where (or whether)
%             the computed level-sets should be plotted. 
%
% OUTPUT: 
%   - Q     : 1-by-K cell containing coordinates of unorderdered line 
%             segments comprising the level-sets.
%   - iv    : 1-by-K array of level-set values.
%   - VT    : 1-by-K cell containg linear interpolation coefficients along 
%             the with indices of mesh vertices that bound the edges 
%             intersected by the computed level-sets.
%
% AUTHOR: Anton Semechko (a.semechko@gmail.com)
%


if nargin<2 || isempty(TR) || isempty(F)
   error('Insufficient number of input arguments') 
end

[Tri,X,fmt]=GetMeshData(TR);

F=F(:);
if ~isnumeric(F) || ~ismatrix(F) || numel(F)~=size(X,1) || sum(isnan(F) | isinf(F))>0
    edit('Invalid entry for 2nd input argument (F)')
end

if nargin<3 || isempty(iv)
    iv=11;
elseif ~isnumeric(iv) || ~ismatrix(iv) || (numel(iv)==1 && (iv<=0 || iv~=round(iv) || isnan(iv) || iv>1001))
    error('Invalid entry for 3rd input argument (iv)')
end

if nargin<4 || isempty(vis)
    vis=false;
elseif numel(vis)~=1 || ~((ishandle(vis) && strcmpi(get(vis,'type'),'axes')) || islogical(vis))
    error('Invalid entry for 4th input argument (vis)')
end

% Level-set values
F_min=min(F);
F_max=max(F);
if numel(iv)==1   
    if iv==1
        iv=(F_max+F_min)/2;
    else
        dF=(F_max-F_min)/iv;
        iv=linspace(F_min+dF/2,F_max-dF/2,iv);
    end
elseif numel(iv)==2 && iv(1)==iv(2)
    iv=iv(1);
end
    
% Edges 
E=[Tri(:,[1 2]);Tri(:,[2 3]);Tri(:,[3 1])];

X1=X(Tri(:,1),:);
X2=X(Tri(:,2),:);
X3=X(Tri(:,3),:);

F1=F(E(:,1)); F1=reshape(F1,[],3);
F2=F(E(:,2)); F2=reshape(F2,[],3);

Fe_min=min(F1,F2);
Fe_max=max(F1,F2);

ha=[];
if islogical(vis) && vis
    
    if fmt>1, TR=triangulation(Tri,X); end
    
    figure('color','w')
    axis equal
    hold on
    h=trimesh(TR);
    set(h,'EdgeColor','none','FaceColor',0.75*[1 1 1],'FaceAlpha',0.75,...
          'SpecularExponent',100,'SpecularStrength',0.25);
    ha=gca;
    h1=camlight('headlight');
    set(h1,'style','infinite','position',10*get(h1,'position'))
    h2=light('position',-get(h1,'position'));
    set(h2,'style','infinite')
    lighting phong
elseif ishandle(vis)
    ha=vis;
    hold on
end

flag=false;
if nargout>2, flag=true; end

% Level-sets
n=numel(iv);
Q=cell(n,1);
VT=cell(n,1);
H=nan(1,n);
for i=1:n % loop through level-sets

    % Edges enclosing i-th level-set
    e_id=Fe_min<=iv(i) & Fe_max>=iv(i);    
    m=sum(e_id,2);
    f_id=m<=2 & m>0; % triangles traversed by the level-set    
    e_id=e_id(f_id,:);
    
    %                        E1      E2      E3
    % Triangle vertices : 1 ----> 2 ----> 3 ----> 1
    X1_i=X1(f_id,:);
    X2_i=X2(f_id,:);
    X3_i=X3(f_id,:);
    v_id=Tri(f_id,:);
    
    % Points of intersection; where level-set crosses the edges
    dFe=F2(f_id,:)-F1(f_id,:);
    t=(iv(i)-F1(f_id,:))./dFe;    

    id_slf=abs(dFe)<=1E-15;
    if sum(id_slf(:))>0; % check if there are edges lying exactly on the level-set
        t(id_slf)=0;
    end
    
    P1=bsxfun(@times,t(:,1),X2_i) + bsxfun(@times,1-t(:,1),X1_i); % E1 
    P2=bsxfun(@times,t(:,2),X3_i) + bsxfun(@times,1-t(:,2),X2_i); % E2
    P3=bsxfun(@times,t(:,3),X1_i) + bsxfun(@times,1-t(:,3),X3_i); % E3
    
    % Line segments comprising the level-set
    id_12=e_id(:,1) & e_id(:,2); % E1 -- E2
    id_13=e_id(:,1) & e_id(:,3); % E1 -- E3
    id_23=e_id(:,2) & e_id(:,3); % E2 -- E3

    
    if sum(id_12)>0        
        P12=cat(3,P1(id_12,:),P2(id_12,:));
        P12(:,:,3)=NaN;
        P12=permute(P12,[3 1 2]); 
        P12=reshape(P12,[],3);
        
        if flag
            V12=cat(3,[v_id(id_12,1),v_id(id_12,2) t(id_12,1)],[v_id(id_12,2),v_id(id_12,3) t(id_12,2)]);
            V12(:,:,3)=NaN;
            V12=permute(V12,[3 1 2]);
            V12=reshape(V12,[],3);
        end
    else
        P12=[]; V12=[];
    end
    
    if sum(id_13)>0
        P13=cat(3,P1(id_13,:),P3(id_13,:));
        P13(:,:,3)=NaN;
        P13=permute(P13,[3 1 2]);
        P13=reshape(P13,[],3);
        
        if flag
            V13=cat(3,[v_id(id_13,1),v_id(id_13,2) t(id_13,1)],[v_id(id_13,3),v_id(id_13,1) t(id_13,3)]);
            V13(:,:,3)=NaN;
            V13=permute(V13,[3 1 2]);
            V13=reshape(V13,[],3);
        end
    else
        P13=[]; V13=[];
    end
    
    if sum(id_23)>0                
        P23=cat(3,P2(id_23,:),P3(id_23,:));
        P23(:,:,3)=NaN;
        P23=permute(P23,[3 1 2]);
        P23=reshape(P23,[],3);
        
        if flag
            V23=cat(3,[v_id(id_23,2),v_id(id_23,3) t(id_23,2)],[v_id(id_23,3),v_id(id_23,1) t(id_23,3)]);
            V23(:,:,3)=NaN;
            V23=permute(V23,[3 1 2]);
            V23=reshape(V23,[],3);
        end
    else
        P23=[]; V23=[];
    end
    P=cat(1,P12,P13,P23);
    
    Q{i}=P;
    if flag, VT{i}=cat(1,V12,V13,V23); end
     
    if ishandle(ha) && ~isempty(P)
        H(i)=plot3(ha,P(:,1),P(:,2),P(:,3),'-k','LineWidth',2);
    end
    
    if isempty(P)
        fprintf(2,'Value %f is outside the domain of F\n',iv(i));
    end
    
end

