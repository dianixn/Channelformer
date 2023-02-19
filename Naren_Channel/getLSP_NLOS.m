function LSP_NLOS = getLSP_NLOS(Fc_GHz,Dis2D_NLOS,UE_heightNLOS,ShaNLOS)
%%% This function is to calculate large-scale parameters(SF,DS,ASD,ASA,ZSD,ZSA) in Step 4.
%%% L is the number of NLOS links.
% Input
%       Fc_GHz                 :            carrier frequency in GHz
%       Dis2D_NLOS             :            2D link length, L x 1
%       UE_heightNLOS          :            UE antenna heights, L x 1
%       ShaNLOS                :            shadow fading std, dB
% Output
%       LSP_NLOS               :            7 x L, each column including SF, DS, ASD, ASA, ZSD, ZSA and mu_zsd of each NLOS link.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumNLOS = length(Dis2D_NLOS); 

% Cross-correlation matrix (Table 7.5-6 Part 1)

SF_DS = -0.4; 
SF_ASD = -0.6;
SF_ASA = 0; 
SF_ZSD = 0;
SF_ZSA = -0.4;
DS_ASD = 0.4;
DS_ASA = 0.6;
DS_ZSD = -0.5;
DS_ZSA = 0;
ASD_ASA = 0.4;
ASD_ZSD = 0.5;
ASD_ZSA = -0.1;
ASA_ZSD = 0;
ASA_ZSA = 0;
ZSD_ZSA = 0;

C_matrix = [1 SF_DS SF_ASD SF_ASA SF_ZSD SF_ZSA;
            SF_DS 1 DS_ASD DS_ASA DS_ZSD DS_ZSA;
            SF_ASD DS_ASD 1 ASD_ASA ASD_ZSD ASD_ZSA;
            SF_ASA DS_ASA ASD_ASA 1 ASA_ZSD ASA_ZSA;
            SF_ZSD DS_ZSD ASD_ZSD ASA_ZSD 1 ZSD_ZSA;
            SF_ZSA DS_ZSA ASD_ZSA ASA_ZSA ZSD_ZSA 1]; % 6 x 6
        
% Square root matrix
C = sqrtm(C_matrix)*randn(length(C_matrix),NumNLOS); % 6 x L 

% Randomly generating LSP (frequency-dependent), 1 x L

% SF
sigma_sf = ShaNLOS; % dB
SF = C(1,:).*sigma_sf; % in dB

% DS
mu_ds = -6.28-0.204*log10(max(Fc_GHz,6)); % Note 6 in Table 7.5-6 Part 1
sigma_ds = 0.39;
DS = 10.^(mu_ds+C(2,:).*sigma_ds); 

% ASD
mu_asd = 1.5-0.1144*log10(max(Fc_GHz,6));
sigma_asd = 0.28;
ASD = 10.^(mu_asd+C(3,:).*sigma_asd);

% ASA
mu_asa = 2.08-0.27*log10(max(Fc_GHz,6));
sigma_asa = 0.11;
ASA = 10.^(mu_asa+C(4,:).*sigma_asa);

% ZSD
mu_zsd = (max(-0.5,-2.1*(Dis2D_NLOS./1000)-0.01.*(UE_heightNLOS-1.5)+0.9)).'; 
sigma_zsd = 0.49*ones(size(mu_zsd));
ZSD = 10.^(mu_zsd+C(5,:).*sigma_zsd);

% ZSA
mu_zsa = -0.3236*log10(max(Fc_GHz,6))+1.512;
sigma_zsa = 0.16;
ZSA = 10.^(mu_zsa+C(6,:).*sigma_zsa);

ASA = min(104,ASA);
ASD = min(104,ASD);
ZSA = min(52,ZSA);
ZSD = min(52,ZSD);

LSP_NLOS = [SF;DS;ASD;ASA;ZSD;ZSA;mu_zsd]; % 7 x L
