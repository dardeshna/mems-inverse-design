clearvars;

syms xi eta zeta

n = 20;

coords = [-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; -1 -1 1; 1 -1 1; 1 1 1; -1 1 1; 0 -1 -1; 1 0 -1; 0 1 -1; -1 0 -1; 0 -1 1; 1 0 1; 0 1 1; -1 0 1; -1 -1 0; 1 -1 0; 1 1 0; -1 1 0];

N = sym('N', [n, 1]);

for i=1:n
    xi_a = coords(i, 1);
    eta_a = coords(i, 2);
    zeta_a = coords(i, 3);
    if all(coords(i,:))
        N(i) = 1/8 * (1 + xi*xi_a)*(1 + eta*eta_a)*(1 + zeta*zeta_a)*(xi*xi_a + eta*eta_a + zeta*zeta_a - 2);
    elseif coords(i,1) == 0
        N(i) = 1/4 * (1 - xi^2)*(1 + eta*eta_a)*(1 + zeta*zeta_a);
    elseif coords(i,2) == 0
        N(i) = 1/4 * (1 + xi*xi_a)*(1 - eta^2)*(1 + zeta*zeta_a);
    elseif coords(i,3) == 0
        N(i) = 1/4 * (1 + xi*xi_a)*(1 + eta*eta_a)*(1 - zeta^2);
    end
end

% for i=1:n
%     
%     [i, coords(i,:), N(i), subs(N(i), [xi, eta, zeta], coords(i,:))]
%     for j=1:n
%         
%         if i==j
%             continue
%             
%         end
%         
%         subs(N(i), [xi, eta, zeta], coords(j,:))
%         
%     end
% 
% end

grad_N = jacobian(N, [xi, eta, zeta])

N_func = matlabFunction(N, 'File', 'N.m')
grad_N_func = matlabFunction(grad_N, 'File', 'grad_N.m')