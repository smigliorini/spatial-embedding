import model0 as m
# CNN NETWORK
a,g,b,r = m.create_h(2500,128,128,6,1,5)
# local histograms
autoenc_a, a_train, a_test = m.auto_encoder(1,1,0,128,128,6,12,128,a,g,b,r)
# global histograms
autoenc_g, g_train, g_test = m.auto_encoder(1,0,1,128,128,6,12,128,a,g,b,r)
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
# plot results for local histograms
import plot as p
p.plot_h6_f(a_test, dec_a_test,0,10,'t1_cnn.png')
p.plot_h6_f(a_test, dec_a_test,100,10, 't2_cnn.png')
p.plot_h6_f(a_test, dec_a_test,250,10, 't3_cnn.png')
p.plot_h6_f(a_test, dec_a_test,350,10, 't4_cnn.png')
# plot results for global histograms
p.plot_h1_f(g_test, dec_g_test,0,10, 'tg1_cnn.png')
p.plot_h1_f(g_test, dec_g_test,100,10, 'tg2_cnn.png')
