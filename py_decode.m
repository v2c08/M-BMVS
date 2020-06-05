function g = py_decode(z)
% g  - visual prediction (\tilde(y) in paper)
% z  - beliefs (v_h, v_c in paper)

    m = getDecoder(); % p_theta
    g = squeeze(double(py.load_model.decode(m, z))); 

end
