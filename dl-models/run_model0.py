import model0 as m
a,g,b,r = m.create_h(2500,128,128,6,1,2)
# local histograms
autoenc_a, a_train, a_test = m.auto_encoder(1,0,128,128,6,1024,128,a,g,b,r)
# global histograms
autoenc_g, g_train, g_test = m.auto_encoder(0,1,128,128,6,1024,128,a,g,b,r)
# enc dec global
enc_g = autoenc_g.encoder(g).numpy()
dec_g = autoenc_g.decoder(enc_g).numpy()
enc_g_pred = autoenc_g.encoder(g_test).numpy()
dec_g_pred = autoenc_g.decoder(enc_g_pred).numpy()
# enc dec local
enc_a = autoenc_a.encoder(a).numpy()
dec_a = autoenc_a.decoder(enc_a).numpy()
enc_a_test = autoenc_a.encoder(a_test).numpy()
dec_a_test = autoenc_a.decoder(enc_a_test).numpy()
# plot the result
import plot as p
p.plot_h6(a_test, dec_a_test,0,10)
p.plot_h6(a_test, dec_a_test,100,10)
